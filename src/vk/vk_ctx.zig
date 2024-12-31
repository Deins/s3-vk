//! Vulkan application context and initialization utilities
//! In general this provides only basics, should be customized for more advanced use cases.
//! VkContext - stores in single struct for convenience all important device handles, instance & device info
//! Declares vulkan api version to use & extensions
const std = @import("std");
const vk = @import("vulkan");
const slog = std.log.scoped(.vk_ctx);
const vk_instance = @import("vk_instance.zig");
const apis = vk_instance.apis;

const required_device_extensions = [_][*:0]const u8{vk.extensions.khr_swapchain.name};

pub const VkContext = struct {
    pub const DeviceProxy = vk.DeviceProxy(apis);
    pub const CommandBuffer = vk.CommandBufferProxy(apis);
    pub const QueueProxy = vk.QueueProxy(apis);

    // const max_queues = 6;
    // const QueueStrategy = {
    //     single,
    //     single_plus_transfer,
    //     separate,
    // };
    // const Queues = union(QueueStrategy) {};

    pub const Options = struct {
        instance: vk_instance.InstanceProxy,
        surface: vk.SurfaceKHR = vk.SurfaceKHR.null_handle,
        selected_device: ?u32 = null, // which gpu device to use
        vk_alloc: ?*vk.AllocationCallbacks = null, // host side vulkan allocator
        main_queue_flags: vk.QueueFlags = .{
            .graphics_bit = true,
            .compute_bit = true,
        },
    };

    alloc: std.mem.Allocator,
    vk_alloc: ?*vk.AllocationCallbacks,
    instance: vk_instance.InstanceProxy,
    dev: DeviceProxy,
    pdev: vk.PhysicalDevice,
    surface: vk.SurfaceKHR,
    cmd_pool: vk.CommandPool,
    cmd_buff: vk.CommandBuffer,

    main_queue: vk.Queue, // gfx + present, possibly compute & transfer but not guaranteed
    main_queue_idx: u32,
    // transfer_queue: ?vk.Queue, // separate transfer queue if available
    // transfer_queue_idx,
    // compute_queue: ?vk.Queue, // separate compute queue if available
    // compute_queue_idx: u32,

    selected_phy_device_idx: usize,
    phy_devices: std.MultiArrayList(VkPhyDevice),

    pub fn init(alloc: std.mem.Allocator, options: Options) !VkContext {
        const vki = options.instance;
        const vk_alloc = options.vk_alloc;

        var phy_device_count: u32 = 0;
        vkHandleErrResult(vki.enumeratePhysicalDevices(&phy_device_count, null)); // get number of devices
        if (phy_device_count <= 0) @panic("No vulkan devices available!");
        var phy_devices: std.MultiArrayList(VkPhyDevice) = .{};
        try phy_devices.resize(alloc, phy_device_count);
        //errdefer phy_devices.deinit(alloc);
        vkHandleErrResult(vki.enumeratePhysicalDevices(&phy_device_count, phy_devices.items(.dev).ptr)); // get device handles
        std.debug.assert(phy_device_count == phy_devices.len);

        // read device properties
        for (phy_devices.items(.dev), 0..) |phy_dev_handle, i| {
            phy_devices.set(i, try VkPhyDevice.init(phy_dev_handle, vki, alloc));
            //const props = phy_devices.items(.props)[i];
            //std.debug.print("Device {}#: '{s}' ({})\n", .{ i, props.device_name, props.device_id });

            // std.debug.print("\textensions:\n", .{});
            // for (phy_devices.items(.extensions)[i]) |extension| std.debug.print("\t\t{s} V{}\n", .{ std.mem.span(@as([*:0]const u8, @ptrCast(&extension.extension_name))), extension.spec_version });

            // std.debug.print("\tqueue_props: {any}\n", .{phy_devices.items(.queue_props)[i]});

            // const fmtSize = std.fmt.fmtIntSizeBin;
            // std.debug.print("\tmemory_types:\n", .{});
            // for (0..phy_devices.items(.mem_props)[i].memory_type_count) |mi| std.debug.print("\t\t{any}\n", .{phy_devices.items(.mem_props)[i].memory_types[mi]});
            // std.debug.print("\tmemory_heaps:\n", .{});
            // for (0..phy_devices.items(.mem_props)[i].memory_heap_count) |mi| std.debug.print("\t\t{} => {any}\n", .{
            //     fmtSize(phy_devices.items(.mem_props)[i].memory_heaps[mi].size),
            //     phy_devices.items(.mem_props)[i].memory_heaps[mi].flags,
            // });
        }

        // TODO: check and select device that support needed features etc..
        const selected_device_idx_ = options.selected_device orelse 0;
        const selected_device_idx = if (selected_device_idx_ < phy_devices.len) selected_device_idx_ else {
            slog.err("Selected vulkan device does not exist ! {}/{}", .{ selected_device_idx_, phy_devices.len });
            return error.NoVkDevice;
        };
        const main_queue_idx: u32 = blk: {
            for (phy_devices.items(.queue_props)[selected_device_idx], 0..) |queue_props, i| {
                if (@as(vk.Flags, @bitCast(queue_props.queue_flags)) & @as(vk.Flags, @bitCast(options.main_queue_flags)) == @as(vk.Flags, @bitCast(options.main_queue_flags))) break :blk @intCast(i);
            }
            std.debug.panic("Selected vulkan device {} has no compute queue", .{selected_device_idx});
        };
        // const compute_queue_idx = blk: {
        //     for (phy_devices.items(.queue_props)[selected_device_idx], 0..) |queue_props, i| if (queue_props.queue_flags.compute_bit) break :blk i;
        //     std.debug.panic("Selected vulkan device {} has no compute queue", .{selected_device_idx});
        // };
        const qci = [_]vk.DeviceQueueCreateInfo{
            .{
                .queue_family_index = @intCast(main_queue_idx),
                .queue_count = 1,
                .p_queue_priorities = &[_]f32{1},
            },
        };
        slog.info("Using device {}#: '{s}'", .{ selected_device_idx, phy_devices.get(selected_device_idx).props.device_name });
        const dev_handle = try vki.createDevice(phy_devices.items(.dev)[selected_device_idx], &.{
            .queue_create_info_count = 1,
            .p_queue_create_infos = qci[0..],
            .enabled_extension_count = required_device_extensions.len,
            .pp_enabled_extension_names = @ptrCast(&required_device_extensions[0]),
        }, null);
        const dev_wrapper = try alloc.create(vk.DeviceWrapper(apis));
        dev_wrapper.* = try vk.DeviceWrapper(apis).load(dev_handle, vki.wrapper.dispatch.vkGetDeviceProcAddr); // instance_dispatch.dispatch.vkGetDeviceProcAddr
        const dev = vk.DeviceProxy(apis).init(dev_handle, dev_wrapper);
        errdefer dev.destroyDevice(vk_alloc); // todo: this should be done earlier to not leak it in case of proxy/loader failure?

        const main_queue = dev.getDeviceQueue(@intCast(main_queue_idx), 0);
        //const compute_queue = dev.getDeviceQueue(@intCast(compute_queue_idx), 0);

        const cmd_pool = try dev.createCommandPool(&.{
            .queue_family_index = @intCast(main_queue_idx),
        }, vk_alloc);
        errdefer dev.destroyCommandPool(cmd_pool, vk_alloc);

        var cmd_buffers: [1]vk.CommandBuffer = undefined;
        try dev.allocateCommandBuffers(&.{
            .command_pool = cmd_pool,
            .level = .primary,
            .command_buffer_count = cmd_buffers.len,
        }, &cmd_buffers);

        return .{
            .alloc = alloc,
            .vk_alloc = vk_alloc,
            .instance = vki,
            .dev = dev,
            .pdev = phy_devices.items(.dev)[selected_device_idx],
            .surface = options.surface,
            .phy_devices = phy_devices,
            .selected_phy_device_idx = selected_device_idx,
            .main_queue_idx = main_queue_idx,
            .main_queue = main_queue,
            // .compute_queue = compute_queue,
            //.compute_queue_idx = @intCast(compute_queue_idx),
            .cmd_pool = cmd_pool,
            .cmd_buff = cmd_buffers[0],
        };
    }

    pub fn deinit(self: *@This()) void {
        for (0..self.phy_devices.len) |phy_dev_idx| self.phy_devices.get(phy_dev_idx).deinit(self.alloc);
        self.phy_devices.deinit(self.alloc);
        self.dev.freeCommandBuffers(self.cmd_pool, 1, @ptrCast(&self.cmd_buff));
        self.dev.destroyCommandPool(self.cmd_pool, self.vk_alloc);
        self.dev.destroyDevice(self.vk_alloc); // todo: this should be done earlier to not leak it in case of proxy/loader failure?
        //self.instance.destroyInstance(self.vk_alloc); // todo: this should be done earlier to not leak it in case of proxy/loader failure?
        self.alloc.destroy(self.dev.wrapper);
        //self.alloc.destroy(self.instance.wrapper);
    }

    pub fn queueProxy(self: *@This(), queue: vk.Queue) QueueProxy {
        return QueueProxy.init(queue, self.dev.wrapper);
    }

    // most basic allocation, from triangle example
    pub fn findMemoryTypeIndex(self: @This(), memory_type_bits: u32, flags: vk.MemoryPropertyFlags) !u32 {
        const mem_props = self.phy_devices.items(.mem_props)[self.selected_phy_device_idx];
        for (mem_props.memory_types[0..mem_props.memory_type_count], 0..) |mem_type, i| {
            if (memory_type_bits & (@as(u32, 1) << @truncate(i)) != 0 and mem_type.property_flags.contains(flags)) {
                return @truncate(i);
            }
        }

        return error.NoSuitableMemoryType;
    }

    pub fn allocate(self: @This(), requirements: vk.MemoryRequirements, flags: vk.MemoryPropertyFlags) !vk.DeviceMemory {
        return try self.dev.allocateMemory(&.{
            .allocation_size = requirements.size,
            .memory_type_index = try self.findMemoryTypeIndex(requirements.memory_type_bits, flags),
        }, null);
    }
};

// const VkQueue = struct {
//     queue: vk.Queue,
//     family: u32,
//     flags: vk.QueueFlags,
// };

const VkPhyDevice = struct {
    dev: vk.PhysicalDevice,
    props: vk.PhysicalDeviceProperties,
    mem_props: vk.PhysicalDeviceMemoryProperties,
    queue_props: []vk.QueueFamilyProperties,
    features: vk.PhysicalDeviceFeatures,
    extensions: []vk.ExtensionProperties,

    pub fn init(dev: vk.PhysicalDevice, vki: anytype, alloc: std.mem.Allocator) !VkPhyDevice {
        var count: u32 = 0;
        vki.getPhysicalDeviceQueueFamilyProperties(dev, &count, null); // get count
        const queue_props = try alloc.alloc(vk.QueueFamilyProperties, count);
        vki.getPhysicalDeviceQueueFamilyProperties(dev, &count, queue_props.ptr); // read

        vkHandleErrResult(vki.enumerateDeviceExtensionProperties(dev, null, &count, null)); // get count
        const extensions = try alloc.alloc(vk.ExtensionProperties, count);
        vkHandleErrResult(vki.enumerateDeviceExtensionProperties(dev, null, &count, extensions.ptr)); // read

        return .{
            .dev = dev,
            .props = vki.getPhysicalDeviceProperties(dev),
            .mem_props = vki.getPhysicalDeviceMemoryProperties(dev),
            .queue_props = queue_props,
            .features = vki.getPhysicalDeviceFeatures(dev),
            .extensions = extensions,
        };
    }

    pub fn deinit(self: @This(), alloc: std.mem.Allocator) void {
        alloc.free(self.queue_props);
        alloc.free(self.extensions);
    }

    pub fn hasCompute() bool {}
};

// helper to check for both error and result
pub fn vkHandleErrResult(err_or_result: anyerror!vk.Result) void {
    const result = err_or_result catch |err| {
        std.debug.panic("Vulkan failed with error: {}", .{err});
    };
    if (result != vk.Result.success) {
        std.debug.panic("Vulkan returned non successful result: {}", .{result});
    }
}
