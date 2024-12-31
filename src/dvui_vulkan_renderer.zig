//! Vulkan dvui renderer
//! Is not a full backend, only implements rendering related backend functions.
//! All non rendering functions get forwarded to base_backend (whatever backend is selected through build zig).
//! Should work with any base backend as long as it doesn't break from not receiving some command or its not rendering in unexpected calls non rendering related calls.
//!
//! * Currently handles only rendering, initial vulkan setup, swapchain & present logic is expected to be implemented by app.
//! * Renderer performs almost no synchronization outside where it must such as texture creation etc. It follows standart vulkan practice that no more than N (specified at init options) frames in flight will happen.
//!   This synchronization must happen at application level (usually when handling swapchain & present).
//!   If this is not enforced then in flight gpu frame data can get overwritten which can lead to undefined behaviour.
const std = @import("std");
const builtin = @import("builtin");
const slog = std.log.scoped(.dvui_vulkan);
const dvui = @import("dvui");
const vk = @import("vulkan");
const Size = dvui.Size;

const vs_spv align(64) = @embedFile("dvui.vert.spv").*;
const fs_spv align(64) = @embedFile("dvui.frag.spv").*;

const Self = @This();
pub const Vertex = dvui.Vertex;

pub const apis: []const vk.ApiInfo = &(.{vk.features.version_1_0});
pub const DeviceProxy = vk.DeviceProxy(apis);
pub const Indice = u16;
pub const invalid_texture: *anyopaque = @ptrFromInt(0xBAD0); //@ptrFromInt(0xFFFF_FFFF);
const enable_breakpoints = false;
// todo: query min_alignment?
// https://vulkan.gpuinfo.org/displaydevicelimit.php?name=minMemoryMapAlignment&platform=all
const vk_alignment = if (builtin.target.os.tag.isDarwin()) 16384 else 4096;

/// initialization options, caller still owns all passed in resources
pub const InitOptions = struct {
    /// vulkan loader entry point for geting Vulkan functions
    /// we are trying to keep this file independant from user code so we can't share api config (unless we make this whole file generic)
    vkGetDeviceProcAddr: vk.PfnGetDeviceProcAddr,

    /// vulkan device
    dev: vk.Device,
    /// vulkan render pass
    render_pass: vk.RenderPass,
    /// queue - used only for texture upload
    queue: vk.Queue,
    /// command pool - used only for texuture upload
    comamnd_pool: vk.CommandPool,
    /// vulkan physical device
    pdev: vk.PhysicalDevice,
    mem_props: vk.PhysicalDeviceMemoryProperties,
    /// optional vulkan host side allocator
    vk_alloc: ?*vk.AllocationCallbacks = null,

    // /// descriptor pool - must have VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT
    // dpool: vk.DescriptorPool,
    //image_count: u32 = 3,
    // msaa_samples: u32 = 0,

    /// How many dvui frames can be in flight
    /// Should be swapchain image count or larger (if more than one begin/end pair is called per app frame)
    /// Otherwise its possible that previous submitted frame data which is still in flight will get overwritten
    max_frames_in_flight: u32,
    // /// Maximum number of indices  that can be submitted in dvui frame (begin/end).
    // /// if any more indices are submitted than they will be ignored and warning will logged
    // /// TODO: figure out if dvui has similar constant, otherwise implement dynamic buffer growth
    // max_indices: u32 = 0xFFFF,
    // max_vertices: u32 = 0xFFFF,

    /// Maximum number of created textures supported (across all overlapping frames)
    max_textures: u32 = 512,

    /// per frame limits
    max_inflight_buffers: u32 = 1024 * 8,

    /// bytes - total host visible memory allocated
    /// used for passing vertex & index buffers to gpu
    /// this size should be large enough to store all in flight drawClippedTriangles() buffers
    host_visible_memory_size: u32 = maxHostVisibleSize(3, 1024 * 32, 1024 * 32),
    pub inline fn maxHostVisibleSize(max_frames_in_flight: u32, max_vertices_per_frame: u32, max_indices_per_frame: u32) u32 {
        return max_frames_in_flight * (max_vertices_per_frame * @sizeOf(dvui.Vertex) + max_indices_per_frame * @sizeOf(Indice));
    }

    // device (gpu) memory allocation strategy
    // used for created images,

    //device_memory_page_size : u32,
};

/// allocation strategy for device (gpu) memory
const ImageAllocStrategy = union(enum) {
    /// user provides proper allocator
    allocator: struct {},
    /// most basic implementation, ok for few images created with backend.createTexture
    /// WARNING: can consume much of or hit vk.maxMemoryAllocationCount limit too many resources are used, see:
    /// https://vulkan.gpuinfo.org/displaydevicelimit.php?name=maxMemoryAllocationCount&platform=all
    allocate_each: void,
};

const Stats = struct {
    // per frame
    draw_calls: u32 = 0,
};

pub const tex_binding = 1; // shader binding slot must match shader
// pub const ubo_binding = 0; // uniform binding slot must match shader
// const Uniform = extern struct {
//     viewport_size: @Vector(2, f32),
// };

// we need stable pointer to this, but its not worth alocating it, so make it global
var g_dev_wrapper: vk.DeviceWrapper(apis) = undefined;

base_backend: *dvui.backend,

// host owned members
dev: DeviceProxy,
pdev: vk.PhysicalDevice,
vk_alloc: ?*vk.AllocationCallbacks,
cmdbuf: vk.CommandBuffer = .null_handle,
gfx_queue: vk.Queue = .null_handle,
cpool: vk.CommandPool = .null_handle,
dpool: vk.DescriptorPool,

// owned by us
samplers: [2]vk.Sampler,
textures: []Texture,
dset_layout: vk.DescriptorSetLayout,
pipeline: vk.Pipeline,
pipeline_layout: vk.PipelineLayout,
render_target: ?*Texture = null,

host_vis_mem_idx: u32,
host_vis_mem: vk.DeviceMemory,
host_vis_coherent: bool,
host_vis_data: []u8,
host_vis_offset: usize = 0, // linearly advaces and wraps to 0 - assumes size is large enough to not overwrite old still in flight data
device_local_mem_idx: u32,

inflight_buffers: []vk.Buffer,
inflight_buffer_offset: usize = 0, // linearly advaces and wraps to 0 - assumes size is large enough to not overwrite old still in flight data

win_extent: vk.Extent2D = undefined,
prev_scissor: vk.Rect2D = undefined,
dummy_texture: Texture = undefined,

const FrameData = struct {
    vtx_buff: vk.Buffer,
    vtx_mem: []u8,
    idx_buff: vk.Buffer,
    idx_mem: []u8,
    /// textures to be destroyed after frames cycle through
    destroy_textures: []*Texture,
};

pub fn init(alloc: std.mem.Allocator, base_backend: *dvui.backend, opt: InitOptions) !Self {
    const dev_handle = opt.dev;
    g_dev_wrapper = try vk.DeviceWrapper(apis).load(dev_handle, opt.vkGetDeviceProcAddr);
    var dev = vk.DeviceProxy(apis).init(dev_handle, &g_dev_wrapper);

    // Memory
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
    const host_visible_mem = try dev.allocateMemory(&.{
        .allocation_size = opt.host_visible_memory_size,
        .memory_type_index = host_vis_mem_type_index,
    }, opt.vk_alloc);
    errdefer dev.freeMemory(host_visible_mem, opt.vk_alloc);
    const device_local_mem_idx: u32 = blk: {
        for (opt.mem_props.memory_types[0..opt.mem_props.memory_type_count], 0..) |mem_type, i|
            if (mem_type.property_flags.device_local_bit and !mem_type.property_flags.host_visible_bit) {
                slog.debug("chosen device local mem: {} {}", .{ i, mem_type });
                break :blk @truncate(i);
            };
        break :blk host_vis_mem_type_index;
    };

    const dpool_sizes = [_]vk.DescriptorPoolSize{
        .{ .type = .combined_image_sampler, .descriptor_count = opt.max_textures },
        .{ .type = .uniform_buffer, .descriptor_count = opt.max_frames_in_flight },
    };
    const dpool = try dev.createDescriptorPool(&.{
        .max_sets = opt.max_textures + opt.max_frames_in_flight,
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

    const inflight_buffers = try alloc.alloc(vk.Buffer, opt.max_inflight_buffers);
    errdefer alloc.free(inflight_buffers);
    for (inflight_buffers) |*b| b.* = vk.Buffer.null_handle;

    const samplers = [_]vk.SamplerCreateInfo{
        .{ // dvui.TextureInterpolation.nearest
            .mag_filter = .nearest,
            .min_filter = .nearest,
            .mipmap_mode = .nearest,
            .address_mode_u = .repeat,
            .address_mode_v = .repeat,
            .address_mode_w = .repeat,
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
            .address_mode_u = .repeat,
            .address_mode_v = .repeat,
            .address_mode_w = .repeat,
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
        .pipeline = pipeline,
        .pipeline_layout = pipeline_layout,
        .host_vis_mem_idx = host_vis_mem_type_index,
        .host_vis_mem = host_visible_mem,
        .host_vis_data = @as([*]u8, @ptrCast((try dev.mapMemory(host_visible_mem, 0, vk.WHOLE_SIZE, .{})).?))[0..opt.host_visible_memory_size],
        .host_vis_coherent = host_coherent,
        .device_local_mem_idx = device_local_mem_idx,
        .inflight_buffers = inflight_buffers,
        .gfx_queue = opt.queue,
        .cpool = opt.comamnd_pool,
    };
    @memset(res.textures, Texture{});
    {
        const pixels = [4]u8{ 255, 255, 255, 255 };
        res.dummy_texture = try res.createAndUplaodTexture(pixels[0..].ptr, 1, 1, .nearest);
    }
    return res;
}

/// to be safe, call queueWaitIdle before destruction
pub fn deinit(self: *Self, alloc: std.mem.Allocator) void {
    self.dev.destroyDescriptorSetLayout(self.dset_layout, self.vk_alloc);
    for (self.inflight_buffers) |b| if (b != .null_handle) self.dev.destroyBuffer(b, self.vk_alloc);
    alloc.free(self.inflight_buffers);
    self.dummy_texture.deinit(self);
    for (self.samplers) |s| self.dev.destroySampler(s, self.vk_alloc);
    self.dev.destroyDescriptorPool(self.dpool, self.vk_alloc);
    for (self.textures) |tex| if (!tex.isNull()) tex.deinit(self);
    alloc.free(self.textures);

    self.dev.destroyPipelineLayout(self.pipeline_layout, self.vk_alloc);
    self.dev.destroyPipeline(self.pipeline, self.vk_alloc);
    self.dev.unmapMemory(self.host_vis_mem);
    self.dev.freeMemory(self.host_vis_mem, self.vk_alloc);
}

pub fn backend(self: *Self) dvui.Backend {
    return dvui.Backend.initWithCustomContext(Self, self, Self);
}

/// Call this command before rendering dvui in each new render-pass to set vulkan resources used for rendering
///  all rendering commands will be in given cmdbuf
///  gfx_queue & cpool is only used for immediate comamnds that need separate queue such as image transfers
pub fn renderPassStarted(self: *Self, cmdbuf: vk.CommandBuffer, gfx_queue: vk.Queue, cpool: vk.CommandPool) void {
    self.cmdbuf = cmdbuf;
    self.gfx_queue = gfx_queue;
    self.cpool = cpool;
}

/// call this after render pass has ended
/// used to finish submitting textures
pub fn renderPassEnded() void {}

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
    _ = arena; // autofix
    self.render_target = null;
    if (self.cmdbuf == .null_handle) @panic("dvui_vulkan_renderer: Command bufer not set before rendering started!");
    //self.base_backend.begin(arena); // call base

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

    const scissor = vk.Rect2D{
        .offset = .{ .x = 0, .y = 0 },
        .extent = extent,
    };
    self.prev_scissor = scissor;
    dev.cmdSetScissor(cmdbuf, 0, 1, @ptrCast(&scissor));

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
    // slog.info("RENDER: {}, {}, {}", .{ vtx.len, idx.len, self.host_vis_offset });

    { // clip / scissor
        const scissor = if (clipr) |c| vk.Rect2D{
            .offset = .{ .x = @intFromFloat(@max(0, c.x)), .y = @intFromFloat(@max(0, c.y)) },
            .extent = .{ .width = @intFromFloat(c.w), .height = @intFromFloat(c.h) },
        } else vk.Rect2D{
            .offset = .{ .x = 0, .y = 0 },
            .extent = self.win_extent,
        };
        if (!std.meta.eql(self.prev_scissor, scissor)) {
            self.prev_scissor = scissor;
            dev.cmdSetScissor(cmdbuf, 0, 1, @ptrCast(&scissor));
        }
    }

    var idx_buff: vk.Buffer = undefined;
    var vtx_buff: vk.Buffer = undefined;
    { // upload vertices & indices
        var modified_ranges: [2]vk.MappedMemoryRange = undefined;
        var mem_offset = self.host_vis_offset;
        { // indices
            const size = @sizeOf(Indice) * idx.len;
            const buf = dev.createBuffer(&.{
                .size = @sizeOf(Vertex) * idx.len,
                .usage = .{ .index_buffer_bit = true },
                .sharing_mode = .exclusive,
            }, self.vk_alloc) catch |err| {
                slog.err("createBuffer: {}", .{err});
                return;
            };
            const mreq = dev.getBufferMemoryRequirements(buf);

            mem_offset = std.mem.alignForward(usize, mem_offset, mreq.alignment);
            if (size > mreq.size) slog.debug("buffer size req larger: {} {}", .{ size, mreq.size });
            if (mem_offset + mreq.size > self.host_vis_data.len) mem_offset = 0;
            dev.bindBufferMemory(buf, self.host_vis_mem, mem_offset) catch |err| {
                slog.err("bindBufferMemory: {}", .{err});
                dev.destroyBuffer(buf, self.vk_alloc);
                return;
            };
            @memcpy(self.host_vis_data[mem_offset..][0..size], std.mem.sliceAsBytes(idx));
            modified_ranges[0] = .{ .memory = self.host_vis_mem, .offset = mem_offset, .size = size };
            mem_offset += size;
            idx_buff = buf;
        }
        { // vertices
            const size = @sizeOf(Vertex) * vtx.len;
            const buf = dev.createBuffer(&.{
                .size = size,
                .usage = .{ .vertex_buffer_bit = true },
                .sharing_mode = .exclusive,
            }, self.vk_alloc) catch |err| {
                slog.err("createBuffer: {}", .{err});
                return;
            };
            const mreq = dev.getBufferMemoryRequirements(buf);

            mem_offset = std.mem.alignForward(usize, mem_offset, mreq.alignment);
            if (size > mreq.size) slog.debug("buffer size req larger: {} {}", .{ size, mreq.size });
            if (mem_offset + mreq.size > self.host_vis_data.len) mem_offset = 0;
            dev.bindBufferMemory(buf, self.host_vis_mem, mem_offset) catch |err| {
                slog.err("bindBufferMemory: {}", .{err});
                dev.destroyBuffer(buf, self.vk_alloc);
                return;
            };
            @memcpy(self.host_vis_data[mem_offset..][0..size], std.mem.sliceAsBytes(vtx));
            modified_ranges[1] = .{ .memory = self.host_vis_mem, .offset = mem_offset, .size = size };
            mem_offset += size;
            vtx_buff = buf;
        }
        self.host_vis_offset = mem_offset;
        if (!self.host_vis_coherent)
            dev.flushMappedMemoryRanges(modified_ranges.len, &modified_ranges) catch |err|
                slog.err("flushMappedMemoryRanges: {}", .{err});
    }
    // store for destruction
    if (self.inflight_buffer_offset + 2 > self.inflight_buffers.len) self.inflight_buffer_offset = 0;
    for (0..2) |i| if (self.inflight_buffers[self.inflight_buffer_offset + i] != .null_handle) dev.destroyBuffer(self.inflight_buffers[self.inflight_buffer_offset + i], self.vk_alloc);
    self.inflight_buffers[self.inflight_buffer_offset] = idx_buff;
    self.inflight_buffers[self.inflight_buffer_offset + 1] = vtx_buff;
    self.inflight_buffer_offset += 2;

    if (@sizeOf(Indice) != 2) unreachable;
    dev.cmdBindIndexBuffer(cmdbuf, idx_buff, 0, .uint16);
    const offset = [_]vk.DeviceSize{0};
    dev.cmdBindVertexBuffers(cmdbuf, 0, 1, @ptrCast(&vtx_buff), &offset);
    var dset: vk.DescriptorSet = if (texture == null or texture.? == invalid_texture) self.dummy_texture.dset else @as(*Texture, @alignCast(@ptrCast(texture))).dset;
    if (texture != null and dset == self.dummy_texture.dset) {
        //slog.warn("rendering using dummy texture", .{});
        if (enable_breakpoints) @breakpoint();
    }
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
}
pub fn textureCreate(self: *Backend, pixels: [*]u8, width: u32, height: u32, interpolation: dvui.enums.TextureInterpolation) *anyopaque {
    // slog.debug("textureCreate {x}", .{@intFromEnum(img)});
    const tex = self.createAndUplaodTexture(pixels, width, height, interpolation) catch |err| {
        if (enable_breakpoints) @breakpoint();
        slog.err("Can't create texture: {}", .{err});
        return invalid_texture;
    };
    for (self.textures) |*out_tex| {
        if (out_tex.isNull()) {
            out_tex.* = tex;
            //slog.debug("textureCreate: {*}", .{out_tex});
            return @ptrCast(out_tex);
        }
    }
    slog.err("textureCreate: All texture slots are full! Texture discarded.", .{});
    tex.deinit(self);
    return invalid_texture;
}
pub fn textureCreateTarget(self: *Backend, width: u32, height: u32, interpolation: dvui.enums.TextureInterpolation) !*anyopaque {
    _ = width; // autofix
    _ = height; // autofix
    _ = interpolation; // autofix
    // self.renderTarget(invalid_texture);
    return @ptrCast(&self.dummy_texture);
    // return try self.base_backend.textureCreateTarget(width, height, interpolation);
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
    _ = self; // autofix
    if (texture == invalid_texture) return;
    // slog.debug("textureDestroy({*})", .{texture});
    // const tex: *Texture = @ptrCast(@alignCast(texture));
    // tex.deinit(self);
    // tex.* = .{};
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
    // regular alpha
    // const pcbas = vk.PipelineColorBlendAttachmentState{
    //     .blend_enable = vk.TRUE,
    //     .src_color_blend_factor = .src_alpha,
    //     .dst_color_blend_factor = .one_minus_src_alpha,
    //     .color_blend_op = .add,
    //     .src_alpha_blend_factor = .one,
    //     .dst_alpha_blend_factor = .zero,
    //     .alpha_blend_op = .add,
    //     .color_write_mask = .{ .r_bit = true, .g_bit = true, .b_bit = true, .a_bit = true },
    // };

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

fn stageToBuffer(
    self: *@This(),
    buf_info: vk.BufferCreateInfo,
    mreq: vk.MemoryRequirements,
    contents: []const u8,
    inflight_free: bool,
) !vk.Buffer {
    const buf = self.dev.createBuffer(&buf_info, self.vk_alloc) catch |err| {
        slog.err("createBuffer: {}", .{err});
        return err;
    };
    const mreq2 = self.dev.getBufferMemoryRequirements(buf);
    var mem_offset = self.host_vis_offset;
    mem_offset = std.mem.alignForward(usize, mem_offset, mreq2.alignment);
    if (mem_offset + mreq2.size > self.host_vis_data.len) mem_offset = 0;
    self.dev.bindBufferMemory(buf, self.host_vis_mem, mem_offset) catch |err| {
        slog.err("bindBufferMemory: {}", .{err});
        self.dev.destroyBuffer(buf, self.vk_alloc);
        return err;
    };
    @memcpy(self.host_vis_data[mem_offset..][0..contents.len], contents);
    if (!self.host_vis_coherent)
        self.dev.flushMappedMemoryRanges(1, &.{.{ .memory = self.host_vis_mem, .offset = mem_offset, .size = mreq.size }}) catch |err|
            slog.err("flushMappedMemoryRanges: {}", .{err});
    mem_offset += mreq.size;
    self.host_vis_offset = mem_offset;

    if (!inflight_free) return buf;
    if (self.inflight_buffer_offset > self.inflight_buffers.len) self.inflight_buffer_offset = 0;
    self.inflight_buffers[self.inflight_buffer_offset] = buf;
    self.inflight_buffer_offset += 1;
    return buf;
}

fn beginSingleTimeCommands(self: *Self) !vk.CommandBuffer {
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

// fn transitionImageLayout(self: *Self, img: vk.Image, format: vk.Format, old_layout: vk.ImageLayout, new_layout: vk.ImageLayout) void {
//     const img_barrier = vk.ImageMemoryBarrier{
//         .src_access_mask = .{},
//         .dst_access_mask = .{ .transfer_write_bit = true },
//         .old_layout = .undefined,
//         .new_layout = .transfer_dst_optimal,
//         .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
//         .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
//         .image = img,
//         .subresource_range = srr,
//     };
//     dev.cmdPipelineBarrier(cmdbuf, .{ .host_bit = true }, .{ .transfer_bit = true }, .{}, 0, null, 0, null, 1, @ptrCast(&img_barrier));
// }

const Texture = struct {
    img: vk.Image = .null_handle,
    img_view: vk.ImageView = .null_handle,
    mem: vk.DeviceMemory = .null_handle,
    dset: vk.DescriptorSet = .null_handle,

    pub const State = enum {
        deinit,
        recycled,
        in_use,
    };

    pub fn isNull(self: @This()) bool {
        return self.dset == .null_handle;
    }

    pub fn deinit(tex: Texture, b: *Backend) void {
        const dev = b.dev;
        const vk_alloc = b.vk_alloc;
        // TODO: descriptor set individual free ?
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
    }, mreq, pixels[0 .. width * height * 4], false);
    defer dev.destroyBuffer(img_staging, self.vk_alloc);

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
        dev.cmdCopyBufferToImage(cmdbuf, img_staging, img, .transfer_dst_optimal, 1, @ptrCast(&buff_img_copy));

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
