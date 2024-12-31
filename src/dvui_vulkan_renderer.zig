//! Experimental Vulkan dvui renderer
//! Its not a full backend, only implements rendering related backend functions.
//! All non rendering related functions get forwarded to base_backend (whatever backend is selected through build zig).
//! Should work with any base backend as long as it doesn't break from not receiving some of the overridden callbacks or its not rendering from non rendering related calls.
//! Tested only with sdl3 base backend.
//!
//! Notes:
//! * Currently app must provide vulkan setup, swapchain & present logic, etc.
//! * Param `frames_in_flight`, very important:
//!   Currently renderer itself performs almost no locking with gpu except when creating new textures. Synchronization  is archived by using frames in flight as sync:
//!   It is assumed that swapchain has limited number of images (not too many) and at worst case CPU can issue only that many frames before it will have no new frames where render to. That is good blocking/sync point.
//!   By using that knowledge we delay any host->gpu resource operations by max_frames_in_flight to be sure there won't be any data races. (for example textureDeletion is delayed to make sure gpu doesn't use it any more)
//!   But at the moment swapchain management is left for application to do, so application must make sure this is enforced and pass in this information as `max_frames_in_flight` during int.
//!   Otherwise gpu frame data can get overwritten while its still being used leading to undefined behaviour.
//! * Memory: all space for vertex & index buffers is preallocated at start requiring setting appropriate limits in options. Requests to render over limit is safe but will lead to draw commands being dropped.
//!   Texture bookkeeping is also preallocated. But images themselves are allocated individually at runtime. Currently 1 image = 1 allocation which is ok for large or few images,
//!   but not great for many smaller images that can eat in max gpu allocation limit. TODO: implement hooks for general purpose allocator
//!   TODO: currently there also seems to be texture leaks from dvui itself, especially when scaling content. Not all textures seem to be destroyed.
//!
const std = @import("std");
const builtin = @import("builtin");
const slog = std.log.scoped(.dvui_vulkan);
const dvui = @import("dvui");
const vk = @import("vulkan");
const Size = dvui.Size;

const vs_spv align(64) = @embedFile("dvui.vert.spv").*;
const fs_spv align(64) = @embedFile("dvui.frag.spv").*;

const Self = @This();

pub const apis: []const vk.ApiInfo = &(.{vk.features.version_1_0});
pub const DeviceProxy = vk.DeviceProxy(apis);
pub const Vertex = dvui.Vertex;
pub const Indice = u16;
pub const invalid_texture: *anyopaque = @ptrFromInt(0xBAD0BAD0); //@ptrFromInt(0xFFFF_FFFF);

// debug flags
const enable_breakpoints = false;
const texture_tracing = false; // tace leaks and usage

/// initialization options, caller still owns all passed in resources
pub const InitOptions = struct {
    /// vulkan loader entry point for getting Vulkan functions
    /// we are trying to keep this file independent from user code so we can't share api config (unless we make this whole file generic)
    vkGetDeviceProcAddr: vk.PfnGetDeviceProcAddr,

    /// vulkan device
    dev: vk.Device,
    /// vulkan render pass
    render_pass: vk.RenderPass,
    /// queue - used only for texture upload,
    /// used here only once during initialization, afterwards texture upload queue must be provided with beginFrame()
    queue: vk.Queue,
    /// command pool - used only for texuture upload
    comamnd_pool: vk.CommandPool,
    /// vulkan physical device
    pdev: vk.PhysicalDevice,
    /// physical device memory properties
    mem_props: vk.PhysicalDeviceMemoryProperties,
    /// optional vulkan host side allocator
    vk_alloc: ?*vk.AllocationCallbacks = null,

    /// How many frames can be in flight in worst case
    /// In simple single window configuration where synchronization happens when presenting should be at least (swapchain image count - 1) or larger.
    max_frames_in_flight: u32,

    /// Maximum number of indices that can be submitted in single frame
    /// Draw requests above this limit will get discarded
    max_indices_per_frame: u32 = 1024 * 128,
    max_vertices_per_frame: u32 = 1024 * 64,

    /// Maximum number of alive textures supported. global (across all overlapping frames)
    /// Note: as this is only book keeping limit it can be set quite high. Real texture memory usage could be more concernig, as well as large allocation count.
    max_textures: u16 = 256,

    /// error and invalid texture handle color
    /// if by any chance renderer runs out of textures or due to other reason fails to create a texture then this color will be used as texture
    error_texture_color: [4]u8 = [4]u8{ 255, 0, 255, 255 }, // default bright pink so its noticeable for debug, can be set to alpha 0 for invisible etc.

    /// if uv coords go out of bounds, how should the sampling behave
    texture_wrap: vk.SamplerAddressMode = .repeat,

    /// bytes - total host visible memory allocated ahead of time
    pub inline fn hostVisibleMemSize(s: @This()) u32 {
        const vtx_space = std.mem.alignForward(u32, s.max_vertices_per_frame * @sizeOf(dvui.Vertex), vk_alignment);
        const idx_space = std.mem.alignForward(u32, s.max_indices_per_frame * @sizeOf(Indice), vk_alignment);
        return s.max_frames_in_flight * (vtx_space + idx_space);
    }
};

/// TODO:
/// allocation strategy for device (gpu) memory
// const ImageAllocStrategy = union(enum) {
//     /// user provides proper allocator
//     allocator: struct {},
//     /// most basic implementation, ok for few images created with backend.createTexture
//     /// WARNING: can consume much of or hit vk.maxMemoryAllocationCount limit too many resources are used, see:
//     /// https://vulkan.gpuinfo.org/displaydevicelimit.php?name=maxMemoryAllocationCount&platform=all
//     allocate_each: void,
// };

/// just simple debug and informative metrics
pub const Stats = struct {
    // per frame
    draw_calls: u16 = 0,
    verts: u32 = 0,
    indices: u32 = 0,
    // global
    textures_alive: u16 = 0,
    textures_mem: usize = 0,
};

// we need stable pointer to this, but its not worth allocating it, so make it global
var g_dev_wrapper: vk.DeviceWrapper(apis) = undefined;

base_backend: *dvui.backend,

// not owned by us:
dev: DeviceProxy,
pdev: vk.PhysicalDevice,
vk_alloc: ?*vk.AllocationCallbacks,
cmdbuf: vk.CommandBuffer = .null_handle,
dpool: vk.DescriptorPool,
gfx_queue: vk.Queue = .null_handle,
cpool: vk.CommandPool = .null_handle,
// gfx_queue & cpool locking
lock_userdata: ?*anyopaque = null, // user defined data that will be returned in callbacks
lockCB: ?*const fn (userdata: ?*anyopaque) void = null,
unlockCB: ?*const fn (userdata: ?*anyopaque) void = null,

// owned by us
samplers: [2]vk.Sampler,
frames: []FrameData,
textures: []Texture,
destroy_textures_offset: u16 = 0,
destroy_textures: []u16,
pipeline: vk.Pipeline,
pipeline_layout: vk.PipelineLayout,
dset_layout: vk.DescriptorSetLayout,
render_target: ?*Texture = null,
current_frame: *FrameData, // points somewhere in frames

win_extent: vk.Extent2D = undefined,
dummy_texture: Texture = undefined, // dummy/null white texture
error_texture: Texture = undefined,

host_vis_mem_idx: u32,
host_vis_mem: vk.DeviceMemory,
host_vis_coherent: bool,
host_vis_data: []u8, // mapped host_vis_mem
//host_vis_offset: usize = 0, // linearly advaces and wraps to 0 - assumes size is large enough to not overwrite old still in flight data
device_local_mem_idx: u32,

vtx_overflow_logged: bool = false,
idx_overflow_logged: bool = false,
stats: Stats = .{}, // just for info / dbg

const FrameData = struct {
    // buffers to host_vis memory
    vtx_buff: vk.Buffer = .null_handle,
    vtx_data: []u8 = &.{},
    vtx_offset: u32 = 0,
    idx_buff: vk.Buffer = .null_handle,
    idx_data: []u8 = &.{},
    idx_offset: u32 = 0,
    /// textures to be destroyed after frames cycle through
    /// offset & len points to backend.destroy_textures[]
    destroy_textures_offset: u16 = 0,
    destroy_textures_len: u16 = 0,

    fn deinit(f: *@This(), b: *Backend) void {
        f.freeTextures(b);
        b.dev.destroyBuffer(f.vtx_buff, b.vk_alloc);
        b.dev.destroyBuffer(f.idx_buff, b.vk_alloc);
    }

    fn reset(f: *@This(), b: *Backend) void {
        f.vtx_offset = 0;
        f.idx_offset = 0;
        f.destroy_textures_offset = b.destroy_textures_offset;
        f.destroy_textures_len = 0;
    }

    fn freeTextures(f: *@This(), b: *Backend) void {
        // free textures
        for (f.destroy_textures_offset..(f.destroy_textures_offset + f.destroy_textures_len)) |i| {
            const tidx = b.destroy_textures[i % b.destroy_textures.len]; // wrap around on overflow
            // just for debug and monitoring
            b.stats.textures_alive -= 1;
            b.stats.textures_mem -= b.dev.getImageMemoryRequirements(b.textures[tidx].img).size;

            //slog.debug("destroy texture {}({x}) | {}", .{ tidx, @intFromPtr(&b.textures[tidx]), b.stats.textures_alive });
            b.textures[tidx].deinit(b);
            b.textures[tidx].img = .null_handle;
            b.textures[tidx].dset = .null_handle;
            b.textures[tidx].img_view = .null_handle;
            b.textures[tidx].mem = .null_handle;
            b.textures[tidx].trace.addAddr(@returnAddress(), "destroy"); // keep tracing
        }
        f.destroy_textures_len = 0;
    }
};

pub fn init(alloc: std.mem.Allocator, base_backend: *dvui.backend, opt: InitOptions) !Self {
    const dev_handle = opt.dev;
    g_dev_wrapper = try vk.DeviceWrapper(apis).load(dev_handle, opt.vkGetDeviceProcAddr);
    var dev = vk.DeviceProxy(apis).init(dev_handle, &g_dev_wrapper);

    // Memory
    // host visible
    var host_coherent: bool = false;
    const host_vis_mem_type_index: u32 = blk: {
        // device local, host visible
        for (opt.mem_props.memory_types[0..opt.mem_props.memory_type_count], 0..) |mem_type, i|
            if (mem_type.property_flags.device_local_bit and mem_type.property_flags.host_visible_bit) {
                host_coherent = mem_type.property_flags.host_coherent_bit;
                slog.debug("chosen host_visible_mem: {} {}", .{ i, mem_type });
                break :blk @truncate(i);
            };
        // not device local
        for (opt.mem_props.memory_types[0..opt.mem_props.memory_type_count], 0..) |mem_type, i|
            if (mem_type.property_flags.host_visible_bit) {
                host_coherent = mem_type.property_flags.host_coherent_bit;
                slog.info("chosen host_visible_mem is NOT device local - Are we running on integrated graphics?", .{});
                slog.debug("chosen host_visible_mem: {} {}", .{ i, mem_type });
                break :blk @truncate(i);
            };
        return error.NoSuitableMemoryType;
    };
    slog.debug("host_vis allocation size: {}", .{std.fmt.fmtIntSizeBin(opt.hostVisibleMemSize())});
    const host_visible_mem = try dev.allocateMemory(&.{
        .allocation_size = opt.hostVisibleMemSize(),
        .memory_type_index = host_vis_mem_type_index,
    }, opt.vk_alloc);
    errdefer dev.freeMemory(host_visible_mem, opt.vk_alloc);
    const host_vis_data = @as([*]u8, @ptrCast((try dev.mapMemory(host_visible_mem, 0, vk.WHOLE_SIZE, .{})).?))[0..opt.hostVisibleMemSize()];
    // device local mem
    const device_local_mem_idx: u32 = blk: {
        for (opt.mem_props.memory_types[0..opt.mem_props.memory_type_count], 0..) |mem_type, i|
            if (mem_type.property_flags.device_local_bit and !mem_type.property_flags.host_visible_bit) {
                slog.debug("chosen device local mem: {} {}", .{ i, mem_type });
                break :blk @truncate(i);
            };
        break :blk host_vis_mem_type_index;
    };
    // Memory sub-allocation into FrameData
    const frames = try alloc.alloc(FrameData, opt.max_frames_in_flight);
    errdefer alloc.free(frames);
    {
        var mem_offset: usize = 0;
        for (frames) |*f| {
            f.* = .{};
            // TODO: on error here cleanup will leak previous initialized frames
            { // vertex buf
                const buf = try dev.createBuffer(&.{
                    .size = @sizeOf(Vertex) * opt.max_vertices_per_frame,
                    .usage = .{ .vertex_buffer_bit = true },
                    .sharing_mode = .exclusive,
                }, opt.vk_alloc);
                errdefer dev.destroyBuffer(buf, opt.vk_alloc);
                const mreq = dev.getBufferMemoryRequirements(buf);
                mem_offset = std.mem.alignForward(usize, mem_offset, mreq.alignment);
                try dev.bindBufferMemory(buf, host_visible_mem, mem_offset);
                f.vtx_data = host_vis_data[mem_offset..][0..mreq.size];
                f.vtx_buff = buf;
                mem_offset += mreq.size;
            }
            { // index buf
                const buf = try dev.createBuffer(&.{
                    .size = @sizeOf(Indice) * opt.max_indices_per_frame,
                    .usage = .{ .index_buffer_bit = true },
                    .sharing_mode = .exclusive,
                }, opt.vk_alloc);
                errdefer dev.destroyBuffer(buf, opt.vk_alloc);
                const mreq = dev.getBufferMemoryRequirements(buf);
                mem_offset = std.mem.alignForward(usize, mem_offset, mreq.alignment);
                try dev.bindBufferMemory(buf, host_visible_mem, mem_offset);
                f.idx_data = host_vis_data[mem_offset..][0..mreq.size];
                f.idx_buff = buf;
                mem_offset += mreq.size;
            }
        }
    }

    // Descriptors
    const extra: u32 = 8; // idk, exact pool sizes returns OutOfPoolMemory slightly too soon, add extra margin
    const dpool_sizes = [_]vk.DescriptorPoolSize{
        .{ .type = .combined_image_sampler, .descriptor_count = opt.max_textures + extra },
        //.{ .type = .uniform_buffer, .descriptor_count = opt.max_frames_in_flight },
    };
    const dpool = try dev.createDescriptorPool(&.{
        .max_sets = opt.max_textures + extra,
        .pool_size_count = dpool_sizes.len,
        .p_pool_sizes = &dpool_sizes,
        .flags = .{ .free_descriptor_set_bit = true },
    }, opt.vk_alloc);
    const dsl = try dev.createDescriptorSetLayout(
        &vk.DescriptorSetLayoutCreateInfo{
            .binding_count = 1,
            .p_bindings = &.{
                // vk.DescriptorSetLayoutBinding{
                //     .binding = ubo_binding,
                //     .descriptor_count = 1,
                //     .descriptor_type = .uniform_buffer,
                //     .stage_flags = .{ .vertex_bit = true },
                // },
                vk.DescriptorSetLayoutBinding{
                    .binding = tex_binding,
                    .descriptor_count = 1,
                    .descriptor_type = .combined_image_sampler,
                    .stage_flags = .{ .fragment_bit = true },
                },
            },
        },
        opt.vk_alloc,
    );

    const pipeline_layout = try dev.createPipelineLayout(&.{
        .flags = .{},
        .set_layout_count = 1,
        .p_set_layouts = @ptrCast(&dsl),
        .push_constant_range_count = 1,
        .p_push_constant_ranges = &.{.{
            .stage_flags = .{ .vertex_bit = true },
            .offset = 0,
            .size = @sizeOf(f32) * 4,
        }},
    }, opt.vk_alloc);
    const pipeline = try createPipeline(&dev, pipeline_layout, opt.render_pass, opt.vk_alloc);

    const samplers = [_]vk.SamplerCreateInfo{
        .{ // dvui.TextureInterpolation.nearest
            .mag_filter = .nearest,
            .min_filter = .nearest,
            .mipmap_mode = .nearest,
            .address_mode_u = opt.texture_wrap,
            .address_mode_v = opt.texture_wrap,
            .address_mode_w = opt.texture_wrap,
            .mip_lod_bias = 0,
            .anisotropy_enable = 0,
            .max_anisotropy = 0,
            .compare_enable = 0,
            .compare_op = .always,
            .min_lod = 0,
            .max_lod = vk.LOD_CLAMP_NONE,
            .border_color = .int_opaque_white,
            .unnormalized_coordinates = 0,
        },
        .{ // dvui.TextureInterpolation.linear
            .mag_filter = .linear,
            .min_filter = .linear,
            .mipmap_mode = .linear,
            .address_mode_u = opt.texture_wrap,
            .address_mode_v = opt.texture_wrap,
            .address_mode_w = opt.texture_wrap,
            .mip_lod_bias = 0,
            .anisotropy_enable = 0,
            .max_anisotropy = 0,
            .compare_enable = 0,
            .compare_op = .always,
            .min_lod = 0,
            .max_lod = vk.LOD_CLAMP_NONE,
            .border_color = .int_opaque_white,
            .unnormalized_coordinates = 0,
        },
    };

    var res: Self = .{
        .base_backend = base_backend,
        .dev = dev,
        // .vk_render_pass = opt.render_pass,
        .dpool = dpool,
        .pdev = opt.pdev,
        .vk_alloc = opt.vk_alloc,

        .dset_layout = dsl,
        .samplers = .{
            try dev.createSampler(&samplers[0], opt.vk_alloc),
            try dev.createSampler(&samplers[1], opt.vk_alloc),
        },
        .textures = try alloc.alloc(Texture, opt.max_textures),
        .destroy_textures = try alloc.alloc(u16, opt.max_textures),
        .pipeline = pipeline,
        .pipeline_layout = pipeline_layout,
        .host_vis_mem_idx = host_vis_mem_type_index,
        .host_vis_mem = host_visible_mem,
        .host_vis_data = host_vis_data,
        .host_vis_coherent = host_coherent,
        .device_local_mem_idx = device_local_mem_idx,
        .gfx_queue = opt.queue,
        .cpool = opt.comamnd_pool,
        .frames = frames,
        .current_frame = &frames[0],
    };
    @memset(res.textures, Texture{});
    res.dummy_texture = try res.createAndUplaodTexture(&[4]u8{ 255, 255, 255, 255 }, 1, 1, .nearest);
    res.error_texture = try res.createAndUplaodTexture(&opt.error_texture_color, 1, 1, .nearest);
    return res;
}

/// for sync safety, better call queueWaitIdle before destruction
pub fn deinit(self: *Self, alloc: std.mem.Allocator) void {
    for (self.frames) |*f| f.deinit(self);
    alloc.free(self.frames);
    for (self.textures, 0..) |tex, i| if (!tex.isNull()) {
        slog.debug("TEXTURE LEAKED {}:\n", .{i});
        tex.trace.dump();
        tex.deinit(self);
    };
    alloc.free(self.textures);
    alloc.free(self.destroy_textures);

    self.dummy_texture.deinit(self);
    self.error_texture.deinit(self);
    for (self.samplers) |s| self.dev.destroySampler(s, self.vk_alloc);

    self.dev.destroyDescriptorPool(self.dpool, self.vk_alloc);
    self.dev.destroyDescriptorSetLayout(self.dset_layout, self.vk_alloc);
    self.dev.destroyPipelineLayout(self.pipeline_layout, self.vk_alloc);
    self.dev.destroyPipeline(self.pipeline, self.vk_alloc);
    self.dev.unmapMemory(self.host_vis_mem);
    self.dev.freeMemory(self.host_vis_mem, self.vk_alloc);
}

pub fn backend(self: *Self) dvui.Backend {
    return dvui.Backend.initWithCustomContext(Self, self, Self);
}

pub const FrameResources = struct {
    cmdbuf: vk.CommandBuffer,
    // queue & cmdpool used for texture creation & transfers
    // if these resources are used from multiple threads lock/unlock callbacks can be set which will be called
    command_pool: vk.CommandPool,
    queue: vk.Queue,
    lock_userdata: ?*anyopaque = null, // user defined data that will be returned in callbacks
    lockCB: ?*const fn (userdata: ?*anyopaque) void = null,
    unlockCB: ?*const fn (userdata: ?*anyopaque) void = null,
};

/// Begins new frame (frame that counts in overlaping / frames-in-flight), provides renderer with its needed data.
///  All rendering commands will be in given cmdbuf. CmdBuf must be in a RenderPass.
///  gfx_queue & cpool is only used for immediate comamnds such as image transfers that need separate cmdbufs
pub fn beginFrame(self: *Self, frame_resources: FrameResources) void {
    // init resources
    self.cmdbuf = frame_resources.cmdbuf;
    self.gfx_queue = frame_resources.queue;
    self.cpool = frame_resources.command_pool;
    self.lock_userdata = frame_resources.lock_userdata;
    self.lockCB = frame_resources.lockCB;
    self.unlockCB = frame_resources.unlockCB;

    // advance frame pointer,
    const current_frame_idx = (@intFromPtr(self.current_frame) - @intFromPtr(self.frames.ptr) + @sizeOf(FrameData)) / @sizeOf(FrameData) % self.frames.len;
    const cf = &self.frames[current_frame_idx];
    self.current_frame = cf;
    cf.freeTextures(self);

    // reset frame data
    self.current_frame.reset(self);
    self.stats.draw_calls = 0;
    self.stats.indices = 0;
    self.stats.verts = 0;
    self.vtx_overflow_logged = false;
    self.idx_overflow_logged = false;
}

/// call this after render pass has ended
/// used to finish submitting textures
// pub fn endFrame() void {}

//
// Backend interface function overrides
//  see: dvui/Backend.zig
//
const Backend = Self;

pub fn nanoTime(self: *Backend) i128 {
    return self.base_backend.nanoTime();
}
pub fn sleep(self: *Backend, ns: u64) void {
    return self.base_backend.sleep(ns);
}

//pub const begin = Override.begin;
pub fn begin(self: *Self, arena: std.mem.Allocator) void {
    self.render_target = null;
    if (self.cmdbuf == .null_handle) @panic("dvui_vulkan_renderer: Command bufer not set before rendering started!");
    // TODO: get rid of this or do it more cleanly
    // very risky as sdl_backend calls renderer, but have no other way to pass through arena
    self.base_backend.begin(arena); // call base

    //std.log.warn("DvuiVKRendderBegin...", .{});
    // self.base_backend.begin(arena);
    //dvui.backend.begin(@ptrCast(@alignCast(self.base_backend)), arena);
    //std.log.warn("...DvuiVKRendderBegin", .{});

    const dev = self.dev;
    const cmdbuf = self.cmdbuf;
    dev.cmdBindPipeline(cmdbuf, .graphics, self.pipeline);

    const win_size = self.windowSize();
    const extent: vk.Extent2D = .{ .width = @intFromFloat(win_size.w), .height = @intFromFloat(win_size.h) };
    self.win_extent = extent;
    const viewport = vk.Viewport{
        .x = 0,
        .y = 0,
        .width = win_size.w,
        .height = win_size.h,
        .min_depth = 0,
        .max_depth = 1,
    };
    dev.cmdSetViewport(cmdbuf, 0, 1, @ptrCast(&viewport));

    const PushConstants = struct {
        view_scale: @Vector(2, f32),
        view_translate: @Vector(2, f32),
    };
    const push_constants = PushConstants{
        .view_scale = .{ 2.0 / win_size.w, 2.0 / win_size.h },
        .view_translate = .{ -1.0, -1.0 },
    };
    dev.cmdPushConstants(cmdbuf, self.pipeline_layout, .{ .vertex_bit = true }, 0, @sizeOf(PushConstants), &push_constants);
}

pub fn end(self: *Backend) void {
    _ = self; // autofix
}
pub fn pixelSize(self: *Backend) Size {
    // return self.base_backend.pixelSize();
    return windowSize(self);
}
pub fn windowSize(self: *Backend) Size {
    return self.base_backend.windowSize();
}
pub fn contentScale(self: *Backend) f32 {
    return self.base_backend.contentScale();
}
pub fn drawClippedTriangles(self: *Backend, texture: ?*anyopaque, vtx: []const Vertex, idx: []const Indice, clipr: ?dvui.Rect) void {
    if (self.render_target != null) return; // TODO: render to textures
    const dev = self.dev;
    const cmdbuf = self.cmdbuf;
    const cf = self.current_frame;
    const vtx_bytes = vtx.len * @sizeOf(Vertex);
    const idx_bytes = idx.len * @sizeOf(Indice);
    if (cf.vtx_data[cf.vtx_offset..].len < vtx_bytes) {
        if (!self.vtx_overflow_logged) slog.warn("vertex buffer out of space", .{});
        self.vtx_overflow_logged = true;
        if (enable_breakpoints) @breakpoint();
        return;
    }
    if (cf.idx_data[cf.idx_offset..].len < idx_bytes) {
        // if only index buffer alone is out of bounds, we could just shrinking it... but meh
        if (!self.idx_overflow_logged) slog.warn("index buffer out of space", .{});
        self.idx_overflow_logged = true;
        if (enable_breakpoints) @breakpoint();
        return;
    }

    { // clip / scissor
        const scissor = if (clipr) |c| vk.Rect2D{
            .offset = .{ .x = @intFromFloat(@max(0, c.x)), .y = @intFromFloat(@max(0, c.y)) },
            .extent = .{ .width = @intFromFloat(c.w), .height = @intFromFloat(c.h) },
        } else vk.Rect2D{
            .offset = .{ .x = 0, .y = 0 },
            .extent = self.win_extent,
        };
        dev.cmdSetScissor(cmdbuf, 0, 1, @ptrCast(&scissor));
    }

    const idx_offset: u32 = cf.idx_offset;
    const vtx_offset: u32 = cf.vtx_offset;
    { // upload indices & vertices
        var modified_ranges: [2]vk.MappedMemoryRange = undefined;
        { // indices
            const dst = cf.idx_data[cf.idx_offset..][0..idx_bytes];
            cf.idx_offset += @intCast(dst.len);
            modified_ranges[0] = .{ .memory = self.host_vis_mem, .offset = @intFromPtr(dst.ptr) - @intFromPtr(self.host_vis_data.ptr), .size = dst.len };
            @memcpy(dst, std.mem.sliceAsBytes(idx));
        }
        { // vertices
            const dst = cf.vtx_data[cf.vtx_offset..][0..vtx_bytes];
            cf.vtx_offset += @intCast(dst.len);
            modified_ranges[1] = .{ .memory = self.host_vis_mem, .offset = @intFromPtr(dst.ptr) - @intFromPtr(self.host_vis_data.ptr), .size = dst.len };
            @memcpy(dst, std.mem.sliceAsBytes(vtx));
        }
        if (!self.host_vis_coherent)
            dev.flushMappedMemoryRanges(modified_ranges.len, &modified_ranges) catch |err|
                slog.err("flushMappedMemoryRanges: {}", .{err});
    }

    if (@sizeOf(Indice) != 2) unreachable;
    dev.cmdBindIndexBuffer(cmdbuf, cf.idx_buff, idx_offset, .uint16);
    dev.cmdBindVertexBuffers(cmdbuf, 0, 1, @ptrCast(&cf.vtx_buff), &[_]vk.DeviceSize{vtx_offset});
    var dset: vk.DescriptorSet = if (texture == null) self.dummy_texture.dset else blk: {
        if (texture.? == invalid_texture) break :blk self.error_texture.dset;
        const tex = @as(*Texture, @alignCast(@ptrCast(texture)));
        if (tex.trace.index < tex.trace.addrs.len / 2 + 1) tex.trace.addAddr(@returnAddress(), "render"); // if trace has some free room, trace this
        break :blk tex.dset;
    };
    dev.cmdBindDescriptorSets(
        cmdbuf,
        .graphics,
        self.pipeline_layout,
        0,
        1,
        @ptrCast(&dset),
        0,
        null,
    );
    dev.cmdDrawIndexed(cmdbuf, @intCast(idx.len), 1, 0, 0, 0);

    self.stats.draw_calls += 1;
    self.stats.verts += @intCast(vtx.len);
    self.stats.indices += @intCast(idx.len);
}
pub fn textureCreate(self: *Backend, pixels: [*]u8, width: u32, height: u32, interpolation: dvui.enums.TextureInterpolation) *anyopaque {
    var slot: usize = undefined;
    const out_tex: *Texture = blk: {
        for (self.textures, 0..) |*out_tex, s| {
            slot = s;
            if (out_tex.isNull()) break :blk out_tex;
        }
        slog.err("textureCreate: All texture slots! Texture discarded.", .{});
        return invalid_texture;
    };
    const tex = self.createAndUplaodTexture(pixels, width, height, interpolation) catch |err| {
        if (enable_breakpoints) @breakpoint();
        slog.err("Can't create texture: {}", .{err});
        return invalid_texture;
    };
    out_tex.* = tex;
    out_tex.trace = Texture.Trace.init;
    out_tex.trace.addAddr(@returnAddress(), "create");

    self.stats.textures_alive += 1;
    self.stats.textures_mem += self.dev.getImageMemoryRequirements(out_tex.img).size;
    //slog.debug("textureCreate {} ({x}) | {}", .{ slot, @intFromPtr(out_tex), self.stats.textures_alive });
    return @ptrCast(out_tex);
}
pub fn textureCreateTarget(self: *Backend, width: u32, height: u32, interpolation: dvui.enums.TextureInterpolation) !*anyopaque {
    _ = self; // autofix
    _ = width; // autofix
    _ = height; // autofix
    _ = interpolation; // autofix
    // self.renderTarget(invalid_texture);
    //return @ptrCast(&self.dummy_texture);
    return error.NotSupported;
}
pub fn textureRead(self: *Backend, texture: *anyopaque, pixels_out: [*]u8, width: u32, height: u32) !void {
    _ = texture; // autofix
    _ = pixels_out; // autofix
    _ = width; // autofix
    _ = height; // autofix
    //slog.debug("textureRead({}, {*}, {}x{})", .{ texture, pixels_out, width, height });
    _ = self; // autofix
    // return try self.base_backend.textureRead(texture, pixels_out, width, height);
}
pub fn textureDestroy(self: *Backend, texture: *anyopaque) void {
    if (texture == invalid_texture) return;
    const dslot = self.destroy_textures_offset;
    self.destroy_textures_offset = (dslot + 1) % @as(u16, @intCast(self.destroy_textures.len));
    if (self.destroy_textures[dslot] != 0xFFFF)
        self.destroy_textures[dslot] = @intCast((@intFromPtr(texture) - @intFromPtr(self.textures.ptr)) / @sizeOf(Texture));
    self.current_frame.destroy_textures_len += 1;
    // slog.debug("schedule destroy texture: {} ({x})", .{ self.destroy_textures[dslot], @intFromPtr(texture) });
}
pub fn renderTarget(self: *Backend, texture: ?*anyopaque) void {
    // slog.debug("renderTarget({?})", .{texture});
    if (texture == null) self.render_target = null;
    self.render_target = @ptrCast(@alignCast(invalid_texture));
    // return self.base_backend.renderTarget(texture);
}
pub fn clipboardText(self: *Backend) error{OutOfMemory}![]const u8 {
    return self.base_backend.clipboardText();
}
pub fn clipboardTextSet(self: *Backend, text: []const u8) error{OutOfMemory}!void {
    return self.base_backend.clipboardTextSet(text);
}
pub fn openURL(self: *Backend, url: []const u8) error{OutOfMemory}!void {
    return self.base_backend.openURL(url);
}
pub fn refresh(self: *Backend) void {
    return self.base_backend.refresh();
}

//
// Private functions
//

fn createPipeline(
    dev: *DeviceProxy,
    layout: vk.PipelineLayout,
    render_pass: vk.RenderPass,
    vk_alloc: ?*vk.AllocationCallbacks,
) DeviceProxy.CreateGraphicsPipelinesError!vk.Pipeline {
    //  NOTE: VK_KHR_maintenance5 (which was promoted to vulkan 1.4) deprecates ShaderModules.
    // todo: check for extension and then enable
    const ext_m5 = false; // VK_KHR_maintenance5
    const vert_shdd = vk.ShaderModuleCreateInfo{
        .code_size = vs_spv.len,
        .p_code = @ptrCast(&vs_spv),
    };
    const frag_shdd = vk.ShaderModuleCreateInfo{
        .code_size = fs_spv.len,
        .p_code = @ptrCast(&fs_spv),
    };
    var pssci = [_]vk.PipelineShaderStageCreateInfo{
        .{
            .stage = .{ .vertex_bit = true },
            .p_name = "main",
            .module = if (ext_m5) null else try dev.createShaderModule(&vert_shdd, vk_alloc),
            .p_next = if (ext_m5) &vert_shdd else null,
        },
        .{
            .stage = .{ .fragment_bit = true },
            //.module = frag,
            .p_name = "main",
            .module = if (ext_m5) null else try dev.createShaderModule(&frag_shdd, vk_alloc),
            .p_next = if (ext_m5) &frag_shdd else null,
        },
    };
    defer if (!ext_m5) for (pssci) |p| if (p.module != .null_handle) dev.destroyShaderModule(p.module, vk_alloc);

    const pvisci = vk.PipelineVertexInputStateCreateInfo{
        .vertex_binding_description_count = VertexBindings.binding_description.len,
        .p_vertex_binding_descriptions = &VertexBindings.binding_description,
        .vertex_attribute_description_count = VertexBindings.attribute_description.len,
        .p_vertex_attribute_descriptions = &VertexBindings.attribute_description,
    };

    const piasci = vk.PipelineInputAssemblyStateCreateInfo{
        .topology = .triangle_list,
        .primitive_restart_enable = vk.FALSE,
    };

    var viewport: vk.Viewport = undefined;
    var scissor: vk.Rect2D = undefined;
    const pvsci = vk.PipelineViewportStateCreateInfo{
        .viewport_count = 1,
        .p_viewports = @ptrCast(&viewport), // set in createCommandBuffers with cmdSetViewport
        .scissor_count = 1,
        .p_scissors = @ptrCast(&scissor), // set in createCommandBuffers with cmdSetScissor
    };

    const prsci = vk.PipelineRasterizationStateCreateInfo{
        .depth_clamp_enable = vk.FALSE,
        .rasterizer_discard_enable = vk.FALSE,
        .polygon_mode = .fill,
        .cull_mode = .{ .back_bit = false },
        .front_face = .clockwise,
        .depth_bias_enable = vk.FALSE,
        .depth_bias_constant_factor = 0,
        .depth_bias_clamp = 0,
        .depth_bias_slope_factor = 0,
        .line_width = 1,
    };

    const pmsci = vk.PipelineMultisampleStateCreateInfo{
        .rasterization_samples = .{ .@"1_bit" = true },
        .sample_shading_enable = vk.FALSE,
        .min_sample_shading = 1,
        .alpha_to_coverage_enable = vk.FALSE,
        .alpha_to_one_enable = vk.FALSE,
    };

    // do premultiplied alpha blending:
    // const pma_blend = c.SDL_ComposeCustomBlendMode(c.SDL_BLENDFACTOR_ONE, c.SDL_BLENDFACTOR_ONE_MINUS_SRC_ALPHA, c.SDL_BLENDOPERATION_ADD, c.SDL_BLENDFACTOR_ONE, c.SDL_BLENDFACTOR_ONE_MINUS_SRC_ALPHA, c.SDL_BLENDOPERATION_ADD);
    const pcbas = vk.PipelineColorBlendAttachmentState{
        .blend_enable = vk.TRUE,
        .src_color_blend_factor = .one,
        .dst_color_blend_factor = .one_minus_src_alpha,
        .color_blend_op = .add,
        .src_alpha_blend_factor = .one,
        .dst_alpha_blend_factor = .one_minus_src_alpha,
        .alpha_blend_op = .add,
        .color_write_mask = .{ .r_bit = true, .g_bit = true, .b_bit = true, .a_bit = true },
    };

    const pcbsci = vk.PipelineColorBlendStateCreateInfo{
        .logic_op_enable = vk.FALSE,
        .logic_op = .copy,
        .attachment_count = 1,
        .p_attachments = @ptrCast(&pcbas),
        .blend_constants = [_]f32{ 0, 0, 0, 0 },
    };

    const dynstate = [_]vk.DynamicState{ .viewport, .scissor };
    const pdsci = vk.PipelineDynamicStateCreateInfo{
        .flags = .{},
        .dynamic_state_count = dynstate.len,
        .p_dynamic_states = &dynstate,
    };

    const gpci = vk.GraphicsPipelineCreateInfo{
        .flags = .{},
        .stage_count = pssci.len,
        .p_stages = &pssci,
        .p_vertex_input_state = &pvisci,
        .p_input_assembly_state = &piasci,
        .p_tessellation_state = null,
        .p_viewport_state = &pvsci,
        .p_rasterization_state = &prsci,
        .p_multisample_state = &pmsci,
        .p_depth_stencil_state = null,
        .p_color_blend_state = &pcbsci,
        .p_dynamic_state = &pdsci,
        .layout = layout,
        .render_pass = render_pass,
        .subpass = 0,
        .base_pipeline_handle = .null_handle,
        .base_pipeline_index = -1,
    };

    var pipeline: vk.Pipeline = undefined;
    _ = try dev.createGraphicsPipelines(
        .null_handle,
        1,
        @ptrCast(&gpci),
        vk_alloc,
        @ptrCast(&pipeline),
    );
    return pipeline;
}

const AllocatedBuffer = struct {
    buf: vk.Buffer,
    mem: vk.DeviceMemory,
};

/// allocates space for staging, creates buffer, and copies content to it
fn stageToBuffer(
    self: *@This(),
    buf_info: vk.BufferCreateInfo,
    contents: []const u8,
) !AllocatedBuffer {
    const buf = self.dev.createBuffer(&buf_info, self.vk_alloc) catch |err| {
        slog.err("createBuffer: {}", .{err});
        return err;
    };
    errdefer self.dev.destroyBuffer(buf, self.vk_alloc);
    const mreq = self.dev.getBufferMemoryRequirements(buf);
    const mem = try self.dev.allocateMemory(&.{ .allocation_size = mreq.size, .memory_type_index = self.host_vis_mem_idx }, self.vk_alloc);
    errdefer self.dev.freeMemory(mem, self.vk_alloc);
    const mem_offset = 0;
    try self.dev.bindBufferMemory(buf, mem, mem_offset);
    const data = @as([*]u8, @ptrCast((try self.dev.mapMemory(mem, mem_offset, vk.WHOLE_SIZE, .{})).?))[0..mreq.size];
    @memcpy(data[0..contents.len], contents);
    if (!self.host_vis_coherent)
        try self.dev.flushMappedMemoryRanges(1, &.{.{ .memory = mem, .offset = mem_offset, .size = mreq.size }});
    return .{ .buf = buf, .mem = mem };
}

fn beginSingleTimeCommands(self: *Self) !vk.CommandBuffer {
    if (self.lockCB) |f| f(self.lock_userdata);
    var cmdbuf: vk.CommandBuffer = undefined;
    self.dev.allocateCommandBuffers(&.{
        .command_pool = self.cpool,
        .level = .primary,
        .command_buffer_count = 1,
    }, @ptrCast(&cmdbuf)) catch |err| {
        if (enable_breakpoints) @breakpoint();
        return err;
    };
    try self.dev.beginCommandBuffer(cmdbuf, &.{
        .flags = .{ .one_time_submit_bit = true },
    });
    return cmdbuf;
}

fn endSingleTimeCommands(self: *Self, cmdbuf: vk.CommandBuffer) !void {
    defer if (self.unlockCB) |f| f(self.lock_userdata);
    try self.dev.endCommandBuffer(cmdbuf);
    const qs = [_]vk.SubmitInfo{.{
        .wait_semaphore_count = 0,
        .p_wait_semaphores = null,
        .p_wait_dst_stage_mask = null,
        .command_buffer_count = 1,
        .p_command_buffers = @ptrCast(&cmdbuf),
        .signal_semaphore_count = 0,
        .p_signal_semaphores = null,
    }};
    try self.dev.queueSubmit(self.gfx_queue, 1, &qs, .null_handle);
    defer self.dev.freeCommandBuffers(self.cpool, 1, @ptrCast(&cmdbuf));
    // TODO: is there better way to sync this than stalling the queue?
    self.dev.queueWaitIdle(self.gfx_queue) catch |err| {
        slog.warn("queueWaitIdle failed: {}", .{err});
    };
}

const Texture = struct {
    img: vk.Image = .null_handle,
    img_view: vk.ImageView = .null_handle,
    mem: vk.DeviceMemory = .null_handle,
    dset: vk.DescriptorSet = .null_handle,

    trace: Trace = undefined,
    const Trace = std.debug.ConfigurableTrace(6, 5, texture_tracing);

    pub fn isNull(self: @This()) bool {
        return self.dset == .null_handle;
    }

    pub fn deinit(tex: Texture, b: *Backend) void {
        const dev = b.dev;
        const vk_alloc = b.vk_alloc;
        dev.freeDescriptorSets(b.dpool, 1, &[_]vk.DescriptorSet{tex.dset}) catch |err| slog.err("Failed to free descriptor set: {}", .{err});
        dev.destroyImageView(tex.img_view, vk_alloc);
        dev.destroyImage(tex.img, vk_alloc);
        dev.freeMemory(tex.mem, vk_alloc);
    }
};

pub fn createAndUplaodTexture(self: *Backend, pixels: [*]const u8, width: u32, height: u32, interpolation: dvui.enums.TextureInterpolation) !Texture {
    const format = vk.Format.r8g8b8a8_unorm;
    const dev = self.dev;
    var cmdbuf = try self.beginSingleTimeCommands();
    defer self.endSingleTimeCommands(cmdbuf) catch unreachable;
    const img: vk.Image = try dev.createImage(&.{
        //.format = .b8g8r8_unorm,
        .image_type = .@"2d",
        .format = format,
        .extent = .{ .width = width, .height = height, .depth = 1 },
        .mip_levels = 1,
        .array_layers = 1,
        .samples = .{ .@"1_bit" = true },
        .tiling = .optimal,
        .usage = .{
            .transfer_dst_bit = true,
            .sampled_bit = true,
        },
        .sharing_mode = .exclusive,
        .initial_layout = .undefined,
    }, self.vk_alloc);
    errdefer dev.destroyImage(img, self.vk_alloc);
    const mreq = dev.getImageMemoryRequirements(img);

    const mem = dev.allocateMemory(&.{
        .allocation_size = mreq.size,
        .memory_type_index = self.device_local_mem_idx,
    }, self.vk_alloc) catch |err| {
        slog.err("Failed to alloc texture mem: {}", .{err});
        return err;
    };
    errdefer dev.freeMemory(mem, self.vk_alloc);
    try dev.bindImageMemory(img, mem, 0);

    const srr = vk.ImageSubresourceRange{
        .aspect_mask = .{ .color_bit = true },
        .base_mip_level = 0,
        .level_count = 1,
        .base_array_layer = 0,
        .layer_count = 1,
    };
    const img_view = try dev.createImageView(&.{
        .flags = .{},
        .image = img,
        .view_type = .@"2d",
        .format = format,
        .components = .{
            .r = .identity,
            .g = .identity,
            .b = .identity,
            .a = .identity,
        },
        .subresource_range = srr,
    }, self.vk_alloc);
    errdefer dev.destroyImageView(img_view, self.vk_alloc);

    var dset: [1]vk.DescriptorSet = undefined;
    dev.allocateDescriptorSets(&.{
        .descriptor_pool = self.dpool,
        .descriptor_set_count = 1,
        .p_set_layouts = @ptrCast(&self.dset_layout),
    }, &dset) catch |err| {
        if (enable_breakpoints) @breakpoint();
        slog.err("Failed to allocate descriptor set: {}", .{err});
        return err;
    };
    const dii = [1]vk.DescriptorImageInfo{.{
        .sampler = self.samplers[@intFromEnum(interpolation)],
        .image_view = img_view,
        .image_layout = .shader_read_only_optimal,
    }};
    const write_dss = [_]vk.WriteDescriptorSet{.{
        .dst_set = dset[0],
        .dst_binding = tex_binding,
        .dst_array_element = 0,
        .descriptor_count = 1,
        .descriptor_type = .combined_image_sampler,
        .p_image_info = &dii,
        .p_buffer_info = undefined,
        .p_texel_buffer_view = undefined,
    }};
    dev.updateDescriptorSets(write_dss.len, &write_dss, 0, null);

    //slog.info("img {}x{}; req size {}", .{ width, height, mreq.size });
    const img_staging = try self.stageToBuffer(.{
        .flags = .{},
        .size = mreq.size,
        .usage = .{ .transfer_src_bit = true },
        .sharing_mode = .exclusive,
    }, pixels[0 .. width * height * 4]);
    defer dev.destroyBuffer(img_staging.buf, self.vk_alloc);
    defer dev.freeMemory(img_staging.mem, self.vk_alloc);

    { // transition image to dst_optimal
        const img_barrier = vk.ImageMemoryBarrier{
            .src_access_mask = .{},
            .dst_access_mask = .{ .transfer_write_bit = true },
            .old_layout = .undefined,
            .new_layout = .transfer_dst_optimal,
            .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
            .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
            .image = img,
            .subresource_range = srr,
        };
        dev.cmdPipelineBarrier(cmdbuf, .{ .host_bit = true, .top_of_pipe_bit = true }, .{ .transfer_bit = true }, .{}, 0, null, 0, null, 1, @ptrCast(&img_barrier));

        try self.endSingleTimeCommands(cmdbuf);
        cmdbuf = try self.beginSingleTimeCommands();
    }
    { // copy staging -> img
        const buff_img_copy = vk.BufferImageCopy{
            .buffer_offset = 0,
            .buffer_row_length = 0,
            .buffer_image_height = 0,
            .image_subresource = .{
                .aspect_mask = .{ .color_bit = true },
                .mip_level = 0,
                .base_array_layer = 0,
                .layer_count = 1,
            },
            .image_offset = .{ .x = 0, .y = 0, .z = 0 },
            .image_extent = .{ .width = width, .height = height, .depth = 1 },
        };
        dev.cmdCopyBufferToImage(cmdbuf, img_staging.buf, img, .transfer_dst_optimal, 1, @ptrCast(&buff_img_copy));

        try self.endSingleTimeCommands(cmdbuf);
        cmdbuf = try self.beginSingleTimeCommands();
    }
    { // transition to read only optimal
        const img_barrier = vk.ImageMemoryBarrier{
            .src_access_mask = .{ .transfer_write_bit = true },
            .dst_access_mask = .{ .shader_read_bit = true },
            .old_layout = .transfer_dst_optimal,
            .new_layout = .shader_read_only_optimal,
            .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
            .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
            .image = img,
            .subresource_range = srr,
        };
        dev.cmdPipelineBarrier(cmdbuf, .{ .transfer_bit = true }, .{ .fragment_shader_bit = true }, .{}, 0, null, 0, null, 1, @ptrCast(&img_barrier));

        try self.endSingleTimeCommands(cmdbuf);
        cmdbuf = try self.beginSingleTimeCommands();
    }

    return Texture{
        .img = img,
        .img_view = img_view,
        .mem = mem,
        .dset = dset[0],
    };
}

const VertexBindings = struct {
    const binding_description = [_]vk.VertexInputBindingDescription{.{
        .binding = 0,
        .stride = @sizeOf(Vertex),
        .input_rate = .vertex,
    }};

    const attribute_description = [_]vk.VertexInputAttributeDescription{
        .{
            .binding = 0,
            .location = 0,
            .format = .r32g32_sfloat,
            .offset = @offsetOf(Vertex, "pos"),
        },
        .{
            .binding = 0,
            .location = 1,
            .format = .r8g8b8a8_unorm,
            .offset = @offsetOf(Vertex, "col"),
        },
        .{
            .binding = 0,
            .location = 2,
            .format = .r32g32_sfloat,
            .offset = @offsetOf(Vertex, "uv"),
        },
    };
};

pub const tex_binding = 1; // shader binding slot must match shader
// pub const ubo_binding = 0; // uniform binding slot must match shader
// const Uniform = extern struct {
//     viewport_size: @Vector(2, f32),
// };

/// device memory min alignment
/// we could query it at runtime, but this is reasonable safe number. We don't use this for anything critical.
/// https://vulkan.gpuinfo.org/displaydevicelimit.php?name=minMemoryMapAlignment&platform=all
const vk_alignment = if (builtin.target.os.tag.isDarwin()) 16384 else 4096;
