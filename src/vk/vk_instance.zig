const std = @import("std");
const vk = @import("vulkan");
const VkDLL = @import("vk_dll.zig");
const slog = std.log.scoped(.vk_instance);

pub const base_feature_version = vk.features.version_1_2;
pub const apis: []const vk.ApiInfo = &(.{
    .{
        .base_commands = .{
            .createInstance = true,
        },
        .instance_commands = .{
            .createDevice = true,
            .enumeratePhysicalDevices = true,
        },
    },
    // vk.features.version_1_0,
    // vk.features.version_1_1,
    // base_feature_version,
    vk.features.version_1_0,

    // Additional extensions:
    vk.extensions.khr_surface,
    vk.extensions.khr_swapchain,
    //vk.extensions.khr_win_32_surface,

    //vk.extensions.khr_cooperative_matrix,
});

pub const InstanceProxy = vk.InstanceProxy(apis);

pub const InstanceOptions = struct {
    app_name: [:0]const u8, // name of your app for debug purposes and driver optimizations
    vk_alloc: ?*vk.AllocationCallbacks = null, // host side vulkan allocator
    instance_extensions: ?[]const [*:0]const u8 = null,
    vkGetInstanceProcAddr: ?vk.PfnGetInstanceProcAddr = null, // pass in or we will try to find it
    vk_instance: ?vk.Instance = null, // use specified instance instead of creating one
};

var is_init: bool = false;
var g_instance_dispatch: ?*vk.InstanceWrapper(apis) = undefined;
var allocator: std.mem.Allocator = undefined;
pub var vk_alloc: ?*vk.AllocationCallbacks = null;

pub fn init(alloc: std.mem.Allocator, options: InstanceOptions) !InstanceProxy {
    if (is_init) @panic("VkInstance: Double init!");
    is_init = true;
    errdefer is_init = false;
    vk_alloc = options.vk_alloc;
    allocator = alloc;

    if (options.vkGetInstanceProcAddr == null) VkDLL.init() catch return error.VulkanDLL;
    errdefer if (options.vkGetInstanceProcAddr == null) VkDLL.deinit();
    const vkGetInstanceProcAddr = options.vkGetInstanceProcAddr orelse VkDLL.lookup(vk.PfnGetInstanceProcAddr, "vkGetInstanceProcAddr") orelse return error.VkHasNoGetProcAddr;
    const vk_app_info: vk.ApplicationInfo = .{
        .p_application_name = options.app_name.ptr,
        .api_version = base_feature_version.version,
        .application_version = 0,
        .engine_version = 0,
    };

    // var exts = try std.ArrayList([*:0]const u8).initCapacity(alloc, 128);
    // defer exts.deinit();
    // if (options.pp_enabled_extension_names) |iex| for (iex[0..options.enabled_extension_count]) |ext| try exts.append(ext);
    //for (exts.items) |ext| slog.debug("VK ext: {s}", .{ext});

    const instance_info = vk.InstanceCreateInfo{
        .p_application_info = &vk_app_info,
        .enabled_extension_count = if (options.instance_extensions) |e| @intCast(e.len) else 0,
        .pp_enabled_extension_names = if (options.instance_extensions) |e| e.ptr else null,
        // .enabled_extension_count = @intCast(exts.items.len),
        // .pp_enabled_extension_names = exts.items.ptr,
    };
    var vkd = try vk.BaseWrapper(apis).load(vkGetInstanceProcAddr);
    const instance_handle = options.vk_instance orelse try vkd.createInstance(&instance_info, options.vk_alloc); // this is raw vulkan instance pointer
    const instance_dispatch = try alloc.create(vk.InstanceWrapper(apis));
    errdefer alloc.destroy(instance_dispatch);
    g_instance_dispatch = instance_dispatch;
    instance_dispatch.* = try vk.InstanceWrapper(apis).load(instance_handle, vkGetInstanceProcAddr); // create wrapper that loads all vk instance functions
    const vki = vk.InstanceProxy(apis).init(instance_handle, instance_dispatch); // ease of use struct that wraps instance_handle together with instance_dispatch so that instance_ptr doesn't need to be passed in every fn
    return vki;
}

pub fn deinit(vk_instance: InstanceProxy) void {
    if (!is_init) @panic("VkInstance: Double deinit!");
    vk_instance.destroyInstance(null);
    if (g_instance_dispatch) |id| allocator.destroy(id); // cache for later
    g_instance_dispatch = null;
}
