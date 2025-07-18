const std = @import("std");
const builtin = @import("builtin");
const slog = std.log.scoped(.main);
const dvui = @import("dvui");

const vk = @import("vulkan");
const VkContext = @import("vk/vk_ctx.zig").VkContext;
const VkInstance = @import("vk/vk_instance.zig");
const Swapchain = @import("vk/swapchain.zig").Swapchain;
const vert_spv align(64) = @embedFile("triangle.vert.spv").*;
const frag_spv align(64) = @embedFile("triangle.frag.spv").*;

const BaseBackend = dvui.backend;
const DvuiVkRenderer = @import("dvui_vulkan_renderer.zig");

const sdl = BaseBackend.c;
const import_sdl_vk = false;
const sdl_vk = if (import_sdl_vk) @cImport({
    @cInclude("SDL3/SDL_vulkan.h");
}) else struct {
    pub const SDL_FunctionPointer = ?*const fn () callconv(.c) void;
    pub const struct_SDL_Window = opaque {};
    pub const SDL_Window = struct_SDL_Window;

    pub const struct_VkInstance_T = opaque {};
    pub const _VkInstance = ?*struct_VkInstance_T;
    pub const struct_VkPhysicalDevice_T = opaque {};
    pub const VkPhysicalDevice = ?*struct_VkPhysicalDevice_T;
    pub const struct_VkSurfaceKHR_T = opaque {};
    pub const VkSurfaceKHR = ?*struct_VkSurfaceKHR_T;
    pub const struct_VkAllocationCallbacks = opaque {};
    pub extern fn SDL_Vulkan_LoadLibrary(path: [*c]const u8) bool;

    pub extern fn SDL_Vulkan_GetInstanceExtensions(count: [*c]u32) [*c]const [*c]const u8;
    pub extern fn SDL_Vulkan_GetVkGetInstanceProcAddr() SDL_FunctionPointer;
    pub extern fn SDL_Vulkan_CreateSurface(window: ?*u64, instance: _VkInstance, allocator: ?*const struct_VkAllocationCallbacks, surface: [*c]VkSurfaceKHR) bool;
    pub extern fn SDL_Vulkan_DestroySurface(instance: _VkInstance, surface: VkSurfaceKHR, allocator: ?*const struct_VkAllocationCallbacks) void;
    pub extern fn SDL_Vulkan_GetPresentationSupport(instance: _VkInstance, physicalDevice: VkPhysicalDevice, queueFamilyIndex: u32) bool;
};

var sleep_when_inactive = false; // stop rendering when there are no visual changes

/// VSync
const preferred_present_mode: []const vk.PresentModeKHR = &[_]vk.PresentModeKHR{
    .immediate_khr,
    .mailbox_khr,
    .fifo_relaxed_khr,
    .fifo_khr,
};

/// how many frames in flight we want
/// NOTE: swapchain image count = prefered_frames_in_flight + 1 (because 1 is being presented and not worked on)
const prefered_frames_in_flight = 2;
/// just in case we don't get `prefered_frames_in_flight` as fallback
/// max frames in flight app can support (in case device requires more than preferred)
const max_frames_in_flight = 3;

var gpa_instance = std.heap.GeneralPurposeAllocator(.{}){};
const gpa = gpa_instance.allocator();

var scale_val: f32 = 1.0;
const show_demo = true;
var show_dialog_outside_frame: bool = false;
const vulkan = true;

const init_w: f32 = 1024;
const init_h: f32 = 720;

pub fn main() !void {
    const alloc = gpa_instance.allocator();
    defer if (gpa_instance.deinit() != .ok) @panic("memory leak!");
    if (@import("builtin").os.tag == .windows) { // optional
        // on windows graphical apps have no console, so output goes to nowhere - attach it manually. related: https://github.com/ziglang/zig/issues/4196
        _ = winapi.AttachConsole(0xFFFFFFFF);
    }
    slog.info("SDL version: {}", .{BaseBackend.getSDLVersion()});

    dvui.Examples.show_demo_window = false;

    var sdl_win: *sdl.SDL_Window = undefined;
    {
        if (sdl.SDL_Init(sdl.SDL_INIT_VIDEO) != true) {
            dvui.log.err("SDL: Couldn't initialize SDL: {s}", .{sdl.SDL_GetError()});
            return error.BackendError;
        }

        var flags = sdl.SDL_WINDOW_HIGH_PIXEL_DENSITY | sdl.SDL_WINDOW_RESIZABLE;
        if (vulkan) flags |= sdl.SDL_WINDOW_VULKAN;
        // flags |= sdl.SDL_WINDOW_INPUT_FOCUS;
        // flags |= sdl.SDL_WINDOW_ALWAYS_ON_TOP;
        // flags |= sdl.SDL_WINDOW_TRANSPARENT;
        sdl_win = sdl.SDL_CreateWindow("Hello Vulkan", @as(c_int, @intFromFloat(init_w)), @as(c_int, @intFromFloat(init_h)), flags) orelse {
            dvui.log.err("SDL: Failed to open window: {s}", .{sdl.SDL_GetError()});
            return error.BackendError;
        };
        // Limit min dimensions, otherwise we will crash if window gets too small.
        // Otherwise we have to handle swapchain creation failure and skipping rendering when swapchain doesn't exist or something like that.
        if (!sdl.SDL_SetWindowMinimumSize(sdl_win, 64, 64)) return errorSDL("Can't enforce min window dimenstions");
    }

    //
    // Vulkan
    //
    var n_ext: u32 = 0;
    const sdl_platform_extensions = sdl_vk.SDL_Vulkan_GetInstanceExtensions(&n_ext);
    //for (0..n_ext) |i| std.debug.print("sdl req extension: {s}\n", .{sdl_platform_extensions[i]});
    const getProcAddr = sdl_vk.SDL_Vulkan_GetVkGetInstanceProcAddr() orelse return errorSDL("Failed to get vulkan VkGetInstanceProcAddr:");
    const vk_instance = try VkInstance.init(alloc, .{
        .app_name = "MyApp",
        .instance_extensions = @ptrCast(sdl_platform_extensions[0..n_ext]),
        .vkGetInstanceProcAddr = @ptrCast(getProcAddr),
    });
    defer VkInstance.deinit(vk_instance);

    var vk_surface: vk.SurfaceKHR = vk.SurfaceKHR.null_handle;
    if (!sdl_vk.SDL_Vulkan_CreateSurface(@alignCast(@ptrCast(sdl_win)), @ptrFromInt(@intFromEnum(vk_instance.handle)), null, @ptrCast(&vk_surface))) return errorSDL("Failed to create vulkan surface");
    defer sdl_vk.SDL_Vulkan_DestroySurface(@ptrFromInt(@intFromEnum(vk_instance.handle)), @ptrFromInt(@intFromEnum(vk_surface)), null);

    var ctx = try VkContext.init(alloc, .{
        .instance = vk_instance,
        .surface = vk_surface,
    });
    defer ctx.deinit();
    std.debug.assert(sdl_vk.SDL_Vulkan_GetPresentationSupport(@ptrFromInt(@intFromEnum(vk_instance.handle)), @ptrFromInt(@intFromEnum(ctx.pdev)), ctx.main_queue_idx));

    {
        const present_modes = try ctx.instance.getPhysicalDeviceSurfacePresentModesAllocKHR(ctx.pdev, ctx.surface, alloc);
        defer alloc.free(present_modes);
        for (present_modes) |p| slog.info("present modes available: {}", .{p});
    }
    var swapchain = Swapchain.init(&ctx, alloc, .{
        .extent = vk.Extent2D{ .width = init_w, .height = init_h },
        .prefered_image_count = prefered_frames_in_flight,
        .preferred_present_mode = preferred_present_mode,
    }) catch |err| {
        @breakpoint();
        std.debug.panic("Can't create swapchain: {}", .{err});
    };
    if (swapchain.swap_images.len > max_frames_in_flight) std.debug.panic("Swapchain gave us more images than we can handle {}/{}!", .{ swapchain.swap_images.len, max_frames_in_flight });
    defer swapchain.deinit();

    const pipeline_layout = try ctx.dev.createPipelineLayout(&.{
        .flags = .{},
        .set_layout_count = 0,
        .p_set_layouts = undefined,
        .push_constant_range_count = 0,
        .p_push_constant_ranges = undefined,
    }, null);
    defer ctx.dev.destroyPipelineLayout(pipeline_layout, null);

    const render_pass = try createRenderPass(&ctx, swapchain);
    defer ctx.dev.destroyRenderPass(render_pass, null);

    const pipeline = try createPipeline(ctx.dev, pipeline_layout, render_pass);
    defer ctx.dev.destroyPipeline(pipeline, null);

    var framebuffers = try createFramebuffers(&ctx, alloc, render_pass, swapchain);
    defer destroyFramebuffers(&ctx, alloc, framebuffers);

    const pool = try ctx.dev.createCommandPool(&.{
        .queue_family_index = ctx.main_queue_idx,
    }, null);
    defer ctx.dev.destroyCommandPool(pool, null);

    const buffer = try ctx.dev.createBuffer(&.{
        .size = @sizeOf(@TypeOf(vertices)),
        .usage = .{ .transfer_dst_bit = true, .vertex_buffer_bit = true },
        .sharing_mode = .exclusive,
    }, null);
    defer ctx.dev.destroyBuffer(buffer, null);
    const mem_reqs = ctx.dev.getBufferMemoryRequirements(buffer);
    const memory = try ctx.allocate(mem_reqs, .{ .device_local_bit = true });
    defer ctx.dev.freeMemory(memory, null);
    try ctx.dev.bindBufferMemory(buffer, memory, 0);

    var pending_cmdbufs = [_]vk.CommandBuffer{.null_handle} ** max_frames_in_flight;
    defer for (pending_cmdbufs) |cmdbuf| if (cmdbuf != .null_handle) ctx.dev.freeCommandBuffers(pool, 1, @ptrCast(&cmdbuf));
    var last_pending_cmdbuf: usize = 0;

    try uploadVertices(&ctx, pool, buffer);

    //
    // DVUI
    //

    // var renderer: *sdl.SDL_Renderer = undefined;
    // { // create vulkan renderer from properties
    //     const rprops = sdl.SDL_CreateProperties();
    //     defer sdl.SDL_DestroyProperties(rprops);
    //     if (!sdl.SDL_SetPointerProperty(rprops, sdl.SDL_PROP_RENDERER_CREATE_WINDOW_POINTER, sdl_win)) return errorSDL("Failed to set property");
    //     if (!sdl.SDL_SetStringProperty(rprops, sdl.SDL_PROP_RENDERER_CREATE_NAME_STRING, "vulkan")) return errorSDL("Failed to set property");
    //     if (!sdl.SDL_SetPointerProperty(rprops, sdl.SDL_PROP_RENDERER_CREATE_VULKAN_INSTANCE_POINTER, @ptrFromInt(@intFromEnum(vk_instance.handle)))) return errorSDL("Failed to set property");
    //     //if (!sdl.SDL_SetPointerProperty(rprops, sdl.SDL_PROP_RENDERER_CREATE_VULKAN_SURFACE_NUMBER, @ptrFromInt(@intFromEnum(vk_surface)))) return errorSDL("Failed to set property");
    //     if (!sdl.SDL_SetPointerProperty(rprops, sdl.SDL_PROP_RENDERER_CREATE_VULKAN_PHYSICAL_DEVICE_POINTER, @ptrFromInt(@intFromEnum(ctx.pdev)))) return errorSDL("Failed to set property");
    //     if (!sdl.SDL_SetPointerProperty(rprops, sdl.SDL_PROP_RENDERER_CREATE_VULKAN_DEVICE_POINTER, @ptrFromInt(@intFromEnum(ctx.dev.handle)))) return errorSDL("Failed to set property");
    //     if (!sdl.SDL_SetNumberProperty(rprops, sdl.SDL_PROP_RENDERER_CREATE_VULKAN_GRAPHICS_QUEUE_FAMILY_INDEX_NUMBER, ctx.main_queue_idx)) return errorSDL("Failed to set property");
    //     if (!sdl.SDL_SetNumberProperty(rprops, sdl.SDL_PROP_RENDERER_CREATE_VULKAN_PRESENT_QUEUE_FAMILY_INDEX_NUMBER, ctx.main_queue_idx)) return errorSDL("Failed to set property");
    //     renderer = sdl.SDL_CreateRendererWithProperties(rprops) orelse return errorSDL("Failed to create renderer");
    // }
    // defer sdl.SDL_DestroyRenderer(renderer);

    const invalid_renderer: *sdl.SDL_Renderer = @ptrFromInt(0xFFFF_FFFF); // we will use vulkan renderer so sdl renderer will be unused - use dummy invalid pointer that should crash if accessed by accident
    var base_backend = BaseBackend.init(sdl_win, invalid_renderer);
    errdefer base_backend.deinit();
    var dvui_vk_backend = try DvuiVkRenderer.init(alloc, &base_backend, .{
        .max_frames_in_flight = max_frames_in_flight,
        .vkGetDeviceProcAddr = ctx.instance.wrapper.dispatch.vkGetDeviceProcAddr.?,
        .dev = ctx.dev.handle,
        .pdev = ctx.pdev,
        .render_pass = render_pass,
        .queue = ctx.main_queue,
        .comamnd_pool = pool,
        .mem_props = ctx.phy_devices.items(.mem_props)[ctx.selected_phy_device_idx],
        // tight limits
        .max_indices_per_frame = 1024 * 96,
        .max_vertices_per_frame = 1024 * 32,
        // test overflow
        // .max_indices_per_frame = 1024 * 32,
        // .max_vertices_per_frame = 1024 * 16,
    }); // init on top of already initialized backend, overrides rendering
    defer dvui_vk_backend.deinit(alloc);
    defer base_backend.deinit(); // deinit base backend first, so that we can still handle textureDestroy etc.

    // init dvui Window (maps onto a single OS window)
    var win = try dvui.Window.init(@src(), gpa, dvui_vk_backend.backend(), .{});
    defer win.deinit();
    win.theme = win.themes.get("Adwaita Dark").?;

    // backend.initial_scale = sdl.SDL_GetDisplayContentScale(sdl.SDL_GetDisplayForWindow(sdl_win));
    // dvui.log.info("SDL3 backend scale {d}", .{backend.initial_scale});

    defer ctx.dev.queueWaitIdle(ctx.main_queue) catch {}; // let gpu finish its work on exit, otherwise we will get validation errors
    var present_state = Swapchain.PresentState.optimal;
    main_loop: while (true) {
        { // Handle resize and outdated swapchain recreation
            var sdl_w: i32 = 0;
            var sdl_h: i32 = 0;
            if (!sdl.SDL_GetWindowSize(sdl_win, &sdl_w, &sdl_h)) return errorSDL("Can't get window size");
            const extent: vk.Extent2D = .{ .width = @intCast(sdl_w), .height = @intCast(sdl_h) };
            if (present_state == .suboptimal or extent.width != swapchain.extent.width or extent.height != swapchain.extent.height) {
                slog.debug("resize framebuffers: {} -> {}", .{ swapchain.extent, extent });
                ctx.dev.queueWaitIdle(ctx.main_queue) catch {}; // let gpu finish its work, so its safe to recreate swapchain
                swapchain.recreate(extent) catch |err| {
                    slog.err("Resize: Failed to recreate swapchain: {}", .{err});
                };

                destroyFramebuffers(&ctx, alloc, framebuffers);
                framebuffers = createFramebuffers(&ctx, alloc, render_pass, swapchain) catch |err| {
                    slog.err("Resize: Failed to recreate framebuffer: {}", .{err});
                    return err;
                };
            }
        }

        //
        // Vulkan rendering setup & draw triangle
        //
        var cmdbuf: vk.CommandBuffer = undefined;
        try ctx.dev.allocateCommandBuffers(&.{
            .command_pool = pool,
            .level = .primary,
            .command_buffer_count = 1,
        }, @ptrCast(&cmdbuf));
        defer {
            // we can't delete command buffer right after submission, as they are pending execution (in flight). So we buffer them and free later
            const i = (last_pending_cmdbuf + 1) % pending_cmdbufs.len;
            if (pending_cmdbufs[i] != .null_handle) ctx.dev.freeCommandBuffers(pool, 1, @ptrCast(&pending_cmdbufs[i]));
            pending_cmdbufs[i] = cmdbuf;
            last_pending_cmdbuf = i;
        }
        try ctx.dev.beginCommandBuffer(cmdbuf, &.{});

        { // begin render-pass & reset viewport
            const clear = vk.ClearValue{
                .color = .{ .float_32 = .{ 0, 0, 0, 0 } },
            };
            const viewport = vk.Viewport{
                .x = 0,
                .y = 0,
                .width = @floatFromInt(swapchain.extent.width),
                .height = @floatFromInt(swapchain.extent.height),
                .min_depth = 0,
                .max_depth = 1,
            };
            const scissor = vk.Rect2D{
                .offset = .{ .x = 0, .y = 0 },
                .extent = swapchain.extent,
            };
            ctx.dev.cmdBeginRenderPass(cmdbuf, &.{
                .render_pass = render_pass,
                .framebuffer = framebuffers[swapchain.image_index],
                .render_area = vk.Rect2D{
                    .offset = .{ .x = 0, .y = 0 },
                    .extent = swapchain.extent,
                },
                .clear_value_count = 1,
                .p_clear_values = @ptrCast(&clear),
            }, .@"inline");
            ctx.dev.cmdSetViewport(cmdbuf, 0, 1, @ptrCast(&viewport));
            ctx.dev.cmdSetScissor(cmdbuf, 0, 1, @ptrCast(&scissor));
        }

        { // draw triangle scene
            ctx.dev.cmdBindPipeline(cmdbuf, .graphics, pipeline);
            const offset = [_]vk.DeviceSize{0};
            ctx.dev.cmdBindVertexBuffers(cmdbuf, 0, 1, @ptrCast(&buffer), &offset);
            ctx.dev.cmdDraw(cmdbuf, vertices.len, 1, 0, 0);
        }

        //
        // Render DVUI
        //
        const stats = dvui_vk_backend.stats; // grab copy of stats
        dvui_vk_backend.beginFrame(cmdbuf, swapchain.extent);
        // beginWait coordinates with waitTime below to run frames only when needed
        const nstime = win.beginWait(base_backend.hasEvent());
        try win.begin(nstime);

        const quit = try base_backend.addAllEvents(&win);
        if (quit) break :main_loop;

        try gui_frame();
        try gui_stats(stats, &dvui_vk_backend);

        const end_micros = try win.end(.{});

        // cursor management
        // if (win.cursorRequestedFloating()) |cursor| {
        //     // cursor is over floating window, dvui sets it
        //     base_backend.setCursor(cursor);
        // } else {
        //     // cursor should be handled by application
        //     base_backend.setCursor(.bad);
        // }
        base_backend.textInputRect(win.textInputRequested());

        //
        // End render and present
        //
        ctx.dev.cmdEndRenderPass(cmdbuf);
        // _ = dvui_vk_backend.endFrame();
        try ctx.dev.endCommandBuffer(cmdbuf);

        present_state = swapchain.present(cmdbuf) catch |err| switch (err) {
            error.OutOfDateKHR => Swapchain.PresentState.suboptimal,
            else => |narrow| return narrow,
        };

        // waitTime and beginWait combine to achieve variable framerates
        if (sleep_when_inactive) {
            const wait_event_micros = win.waitTime(end_micros, null);
            base_backend.waitEventTimeout(wait_event_micros);
        }

        // Example of how to show a dialog from another thread (outside of win.begin/win.end)
        // if (show_dialog_outside_frame) {
        //     show_dialog_outside_frame = false;
        //     try dvui.dialog(@src(), .{ .window = &win, .modal = false, .title = "Dialog from Outside", .message = "This is a non modal dialog that was created outside win.begin()/win.end(), usually from another thread." });
        // }
    }
}

// both dvui and SDL drawing
fn gui_frame() !void {
    {
        var m = try dvui.menu(@src(), .horizontal, .{ .background = true, .expand = .horizontal });
        defer m.deinit();

        _ = try dvui.checkbox(@src(), &dvui.Examples.show_demo_window, "show demo", .{ .gravity_x = 0.1 });
        _ = try dvui.checkbox(@src(), &sleep_when_inactive, "sleep when inactive", .{ .gravity_x = 0.1 });

        // var choice: usize = 0;
        // _ = try dvui.dropdown(@src(), &.{ "immediate (no vsync)", "fifo", "mailbox" }, &choice, .{});
    }
    // look at demo() for examples of dvui widgets, shows in a floating window
    try dvui.Examples.demo();
}

fn gui_stats(stats: DvuiVkRenderer.Stats, vk_backend: *DvuiVkRenderer) !void {
    var m = try dvui.box(@src(), .vertical, .{ .background = true, .expand = null, .gravity_x = 1.0, .min_size_content = .{ .w = 300, .h = 0 } });
    defer m.deinit();
    var prc: f32 = 0; // progress bar percent [0..1]

    try dvui.labelNoFmt(@src(), "DVUI VK Backend stats", .{ .expand = .horizontal, .gravity_x = 0.5, .font_style = .heading });
    try dvui.label(@src(), "draw_calls:  {}", .{stats.draw_calls}, .{ .expand = .horizontal });

    const idx_max = vk_backend.current_frame.idx_data.len / @sizeOf(DvuiVkRenderer.Indice);
    try dvui.label(@src(), "indices: {} / {}", .{ stats.indices, idx_max }, .{ .expand = .horizontal });
    prc = @as(f32, @floatFromInt(stats.indices)) / @as(f32, @floatFromInt(idx_max));
    try dvui.progress(@src(), .{ .percent = prc }, .{ .expand = .horizontal, .color_accent = .{ .color = dvui.Color.fromHSLuv(@max(12, (1 - prc * prc) * 155), 99, 50, 100) } });

    const verts_max = vk_backend.current_frame.vtx_data.len / @sizeOf(DvuiVkRenderer.Vertex);
    try dvui.label(@src(), "vertices:  {} / {}", .{ stats.verts, verts_max }, .{ .expand = .horizontal });
    prc = @as(f32, @floatFromInt(stats.verts)) / @as(f32, @floatFromInt(verts_max));
    try dvui.progress(@src(), .{ .percent = prc }, .{ .expand = .horizontal, .color_accent = .{ .color = dvui.Color.fromHSLuv(@max(12, (1 - prc * prc) * 155), 99, 50, 100) } });

    try dvui.label(@src(), "Textures:", .{}, .{ .expand = .horizontal, .font_style = .caption_heading });
    try dvui.label(@src(), "count:  {}", .{stats.textures_alive}, .{ .expand = .horizontal });
    try dvui.label(@src(), "mem (gpu): {:.1}", .{std.fmt.fmtIntSizeBin(stats.textures_mem)}, .{ .expand = .horizontal });

    try dvui.label(@src(), "Static/Preallocated memory (gpu):", .{}, .{ .expand = .horizontal, .font_style = .caption_heading });
    const prealloc_mem = vk_backend.host_vis_data.len;
    try dvui.label(@src(), "total:  {:.1}", .{std.fmt.fmtIntSizeBin(prealloc_mem)}, .{ .expand = .horizontal });
    const prealloc_mem_frame = prealloc_mem / vk_backend.frames.len;
    const prealloc_mem_frame_used = stats.indices * @sizeOf(DvuiVkRenderer.Indice) + stats.verts * @sizeOf(DvuiVkRenderer.Vertex);
    try dvui.label(@src(), "current frame:  {:.1} / {:.1}", .{ std.fmt.fmtIntSizeBin(prealloc_mem_frame_used), std.fmt.fmtIntSizeBin(prealloc_mem_frame) }, .{ .expand = .horizontal });
    prc = @as(f32, @floatFromInt(prealloc_mem_frame_used)) / @as(f32, @floatFromInt(prealloc_mem_frame));
    try dvui.progress(@src(), .{ .percent = prc }, .{ .expand = .horizontal, .color_accent = .{ .color = dvui.Color.fromHSLuv(@max(12, (1 - prc * prc) * 155), 99, 50, 100) } });
}

fn createRenderPass(gc: *VkContext, swapchain: Swapchain) !vk.RenderPass {
    const color_attachment = vk.AttachmentDescription{
        .format = swapchain.surface_format.format,
        .samples = .{ .@"1_bit" = true },
        .load_op = .clear,
        .store_op = .store,
        .stencil_load_op = .dont_care,
        .stencil_store_op = .dont_care,
        .initial_layout = .undefined,
        .final_layout = .present_src_khr,
    };

    const color_attachment_ref = vk.AttachmentReference{
        .attachment = 0,
        .layout = .color_attachment_optimal,
    };

    const subpass = vk.SubpassDescription{
        .pipeline_bind_point = .graphics,
        .color_attachment_count = 1,
        .p_color_attachments = @ptrCast(&color_attachment_ref),
    };

    // TODO: review, this is needed for texture creation barriers otherwise validation is angry
    //  Should dvui_renderer create its own render-pass / subpass so that user level doesn't need to be in sync?
    const dep = vk.SubpassDependency{
        .src_subpass = 0, // Self-dependency for subpass 0
        .dst_subpass = 0, // Self-dependency for subpass 0
        .src_stage_mask = .{
            .color_attachment_output_bit = true,
            .transfer_bit = true,
        },
        .dst_stage_mask = .{
            .color_attachment_output_bit = true,
            .transfer_bit = true,
        },
        .src_access_mask = .{
            .color_attachment_write_bit = true,
            .host_write_bit = true,
        },
        .dst_access_mask = .{ .color_attachment_read_bit = true, .color_attachment_write_bit = true, .transfer_write_bit = true },
        .dependency_flags = .{ .by_region_bit = true },
    };
    _ = dep; // autofix

    return try gc.dev.createRenderPass(&.{
        .attachment_count = 1,
        .p_attachments = @ptrCast(&color_attachment),
        .subpass_count = 1,
        .p_subpasses = @ptrCast(&subpass),
        .dependency_count = 0,
        .p_dependencies = null, //@ptrCast(&dep),
    }, null);
}

fn createPipeline(
    dev: VkContext.DeviceProxy, //VkContext.DeviceProxy,
    layout: vk.PipelineLayout,
    render_pass: vk.RenderPass,
) !vk.Pipeline {
    const vert = try dev.createShaderModule(&.{
        .code_size = vert_spv.len,
        .p_code = @ptrCast(&vert_spv),
    }, null);
    defer dev.destroyShaderModule(vert, null);

    const frag = try dev.createShaderModule(&.{
        .code_size = frag_spv.len,
        .p_code = @ptrCast(&frag_spv),
    }, null);
    defer dev.destroyShaderModule(frag, null);

    const pssci = [_]vk.PipelineShaderStageCreateInfo{
        .{
            .stage = .{ .vertex_bit = true },
            .module = vert,
            .p_name = "main",
        },
        .{
            .stage = .{ .fragment_bit = true },
            .module = frag,
            .p_name = "main",
        },
    };

    const pvisci = vk.PipelineVertexInputStateCreateInfo{
        .vertex_binding_description_count = 1,
        .p_vertex_binding_descriptions = @ptrCast(&Vertex.binding_description),
        .vertex_attribute_description_count = Vertex.attribute_description.len,
        .p_vertex_attribute_descriptions = &Vertex.attribute_description,
    };

    const piasci = vk.PipelineInputAssemblyStateCreateInfo{
        .topology = .triangle_list,
        .primitive_restart_enable = vk.FALSE,
    };

    const pvsci = vk.PipelineViewportStateCreateInfo{
        .viewport_count = 1,
        .p_viewports = undefined, // set in createCommandBuffers with cmdSetViewport
        .scissor_count = 1,
        .p_scissors = undefined, // set in createCommandBuffers with cmdSetScissor
    };

    const prsci = vk.PipelineRasterizationStateCreateInfo{
        .depth_clamp_enable = vk.FALSE,
        .rasterizer_discard_enable = vk.FALSE,
        .polygon_mode = .fill,
        .cull_mode = .{ .back_bit = true },
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

    const pcbas = vk.PipelineColorBlendAttachmentState{
        .blend_enable = vk.FALSE,
        .src_color_blend_factor = .one,
        .dst_color_blend_factor = .zero,
        .color_blend_op = .add,
        .src_alpha_blend_factor = .one,
        .dst_alpha_blend_factor = .zero,
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
        .stage_count = 2,
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
        null,
        @ptrCast(&pipeline),
    );
    return pipeline;
}

fn createFramebuffers(gc: *const VkContext, allocator: std.mem.Allocator, render_pass: vk.RenderPass, swapchain: Swapchain) ![]vk.Framebuffer {
    const framebuffers = try allocator.alloc(vk.Framebuffer, swapchain.swap_images.len);
    errdefer allocator.free(framebuffers);

    var i: usize = 0;
    errdefer for (framebuffers[0..i]) |fb| gc.dev.destroyFramebuffer(fb, null);

    for (framebuffers) |*fb| {
        fb.* = try gc.dev.createFramebuffer(&.{
            .render_pass = render_pass,
            .attachment_count = 1,
            .p_attachments = @ptrCast(&swapchain.swap_images[i].view),
            .width = swapchain.extent.width,
            .height = swapchain.extent.height,
            .layers = 1,
        }, null);
        i += 1;
    }

    return framebuffers;
}

fn destroyFramebuffers(gc: *const VkContext, allocator: std.mem.Allocator, framebuffers: []const vk.Framebuffer) void {
    for (framebuffers) |fb| gc.dev.destroyFramebuffer(fb, null);
    allocator.free(framebuffers);
}

const Vertex = struct {
    const binding_description = vk.VertexInputBindingDescription{
        .binding = 0,
        .stride = @sizeOf(Vertex),
        .input_rate = .vertex,
    };

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
            .format = .r32g32b32_sfloat,
            .offset = @offsetOf(Vertex, "color"),
        },
    };

    pos: [2]f32,
    color: [3]f32,
};

const vertices = [_]Vertex{
    .{ .pos = .{ 0, -0.5 }, .color = .{ 1, 0, 0 } },
    .{ .pos = .{ 0.5, 0.5 }, .color = .{ 0, 1, 0 } },
    .{ .pos = .{ -0.5, 0.5 }, .color = .{ 0, 0, 1 } },
};

fn uploadVertices(gc: *const VkContext, pool: vk.CommandPool, buffer: vk.Buffer) !void {
    const staging_buffer = try gc.dev.createBuffer(&.{
        .size = @sizeOf(@TypeOf(vertices)),
        .usage = .{ .transfer_src_bit = true },
        .sharing_mode = .exclusive,
    }, null);
    defer gc.dev.destroyBuffer(staging_buffer, null);
    const mem_reqs = gc.dev.getBufferMemoryRequirements(staging_buffer);
    const staging_memory = try gc.allocate(mem_reqs, .{ .host_visible_bit = true, .host_coherent_bit = true });
    defer gc.dev.freeMemory(staging_memory, null);
    try gc.dev.bindBufferMemory(staging_buffer, staging_memory, 0);

    {
        const data = try gc.dev.mapMemory(staging_memory, 0, vk.WHOLE_SIZE, .{});
        defer gc.dev.unmapMemory(staging_memory);

        const gpu_vertices: [*]Vertex = @ptrCast(@alignCast(data));
        @memcpy(gpu_vertices, vertices[0..]);
    }

    try copyBuffer(gc, pool, buffer, staging_buffer, @sizeOf(@TypeOf(vertices)));
}

fn copyBuffer(gc: *const VkContext, pool: vk.CommandPool, dst: vk.Buffer, src: vk.Buffer, size: vk.DeviceSize) !void {
    var cmdbuf_handle: vk.CommandBuffer = undefined;
    try gc.dev.allocateCommandBuffers(&.{
        .command_pool = pool,
        .level = .primary,
        .command_buffer_count = 1,
    }, @ptrCast(&cmdbuf_handle));
    defer gc.dev.freeCommandBuffers(pool, 1, @ptrCast(&cmdbuf_handle));

    const cmdbuf = VkContext.CommandBuffer.init(cmdbuf_handle, gc.dev.wrapper);

    try cmdbuf.beginCommandBuffer(&.{
        .flags = .{ .one_time_submit_bit = true },
    });

    const region = vk.BufferCopy{
        .src_offset = 0,
        .dst_offset = 0,
        .size = size,
    };
    cmdbuf.copyBuffer(src, dst, 1, @ptrCast(&region));

    try cmdbuf.endCommandBuffer();

    const si = vk.SubmitInfo{
        .command_buffer_count = 1,
        .p_command_buffers = (&cmdbuf.handle)[0..1],
        .p_wait_dst_stage_mask = undefined,
    };
    try gc.dev.queueSubmit(gc.main_queue, 1, @ptrCast(&si), .null_handle);
    try gc.dev.queueWaitIdle(gc.main_queue);
}

fn destroyCommandBuffers(gc: *const VkContext, pool: vk.CommandPool, allocator: std.mem.Allocator, cmdbufs: []vk.CommandBuffer) void {
    gc.dev.freeCommandBuffers(pool, @truncate(cmdbufs.len), cmdbufs.ptr);
    allocator.free(cmdbufs);
}

pub const std_options: std.Options = .{
    .log_level = .debug,
    .log_scope_levels = &.{.{ .scope = .dvui, .level = .info }},
};

pub fn errorSDL(err: []const u8) !void {
    slog.err("SDL: {s}: {s}", .{ err, sdl.SDL_GetError() });
    @breakpoint();
    return error.SDL;
}

test {
    std.testing.refAllDeclsRecursive(@This());
}

test "simple test" {
    var list = std.ArrayList(i32).init(std.testing.allocator);
    defer list.deinit(); // try commenting this out and see if zig detects the memory leak!
    try list.append(42);
    try std.testing.expectEqual(@as(i32, 42), list.pop());
}

// Optional: windows os only
const winapi = if (builtin.os.tag == .windows) struct {
    extern "kernel32" fn AttachConsole(dwProcessId: std.os.windows.DWORD) std.os.windows.BOOL;
} else struct {};
