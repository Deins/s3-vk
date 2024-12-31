///! utility that tries to find vulkan dll or shared object at runtime instead of link time
///! On linux requires linking libc
/// alternative when linking with vulkan/sdk at compile time:
/// extern fn vkGetInstanceProcAddr(instance: vk.Instance, p_name: [*:0]const u8) callconv(vk.vulkan_call_conv) vk.PfnVoidFunction;
const std = @import("std");
const builtin = @import("builtin");
const slog = std.log.scoped(.vk_dll);
const vk = @import("vulkan");

pub const LibVulkan = if (builtin.target.os.tag == .linux) *anyopaque else std.DynLib;

pub var libvulkan: ?LibVulkan = null;
var vk_get_proc_addr: ?vk.PfnGetInstanceProcAddr = null;

pub fn init() !void {
    if (libvulkan != null) return;
    switch (builtin.target.os.tag) {
        .linux => {
            // use c lib dlopen instead of std.DynLib due to: https://github.com/ziglang/zig/issues/5360
            const path = "libvulkan.so";
            const RTLD_LAZY = 0x00001;
            libvulkan = @ptrCast(dlopen(path, RTLD_LAZY));
            if (libvulkan == null) {
                slog.err("dlopen: {s}", .{dlerror() orelse "<UNKNOWN>"});
                return error.DLOpenFailed;
            }
        },
        .windows => {
            const path = "vulkan-1.dll";
            libvulkan = try std.DynLib.open(path);
        },
        else => @compileError("Platform/OS not implemented"),
    }
}

pub fn deinit() void {
    if (libvulkan == null and vk_get_proc_addr == null) return;
    switch (builtin.target.os.tag) {
        .linux => _ = dlclose(libvulkan.?),
        else => libvulkan.?.close(),
    }
    libvulkan = null;
    vk_get_proc_addr = null;
}

pub fn lookup(T: type, name: [:0]const u8) ?T {
    // see vkGetInstanceProcAddrWrapper comments
    if (std.mem.eql(u8, name, "vkGetInstanceProcAddr")) {
        if (vk_get_proc_addr == null) vk_get_proc_addr = rawLookup(vk.PfnGetInstanceProcAddr, name) orelse return null;
        return &vkGetInstanceProcAddrWrapper;
    }
    return rawLookup(T, name);
}

fn rawLookup(T: type, symbol: [:0]const u8) ?T {
    return switch (builtin.target.os.tag) {
        .linux => @as(T, @ptrCast(dlsym(libvulkan, symbol) orelse return null)),
        else => libvulkan.?.lookup(T, symbol) orelse null,
    };
}

fn vkGetInstanceProcAddrWrapper(instance: vk.Instance, p_name: [*:0]const u8) callconv(vk.vulkan_call_conv) vk.PfnVoidFunction {
    // Make sure when asked to get itself, we return itself not null as can happen due to:
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkGetInstanceProcAddr.html
    // 5. Starting with Vulkan 1.2, vkGetInstanceProcAddr can resolve itself with a NULL instance pointer.
    // this behaviour can be observed on windows WSL2 running ubuntu for example
    if (std.mem.eql(u8, std.mem.span(p_name), "vkGetInstanceProcAddr")) return @ptrCast(&vkGetInstanceProcAddrWrapper);

    const res = vk_get_proc_addr.?(instance, p_name);
    if (res == null) slog.debug("vkGetInstanceProcAddr: can't load '{s}'", .{p_name});
    return res;
}

const dlerror = std.c.dlerror;
const dlopen = std.c.dlopen;
const dlsym = std.c.dlsym;
const dlclose = std.c.dlclose;

// I don't think vulkan spec actually requires recursive calls to return same pointer, but we enforce it and it makes good test case
test "vk_dll_get_proc_itself" {
    try init();
    defer deinit();
    const vkGetProcAddr = lookup(vk.PfnGetInstanceProcAddr, "vkGetInstanceProcAddr");
    const vkGetProcAddr_1: vk.PfnGetInstanceProcAddr = @ptrCast(vkGetProcAddr.?(vk.Instance.null_handle, "vkGetInstanceProcAddr").?);
    const vkGetProcAddr_2 = vkGetProcAddr_1(vk.Instance.null_handle, "vkGetInstanceProcAddr");
    try std.testing.expectEqual(@intFromPtr(vkGetProcAddr), @intFromPtr(vkGetProcAddr_1));
    try std.testing.expectEqual(@intFromPtr(vkGetProcAddr_1), @intFromPtr(vkGetProcAddr_2));
}

test "vk_double_init" {
    try init();
    deinit();
    try init();
    deinit();
}

test "vk_multi_init" {
    try init();
    try init();
    deinit();
    deinit();
}
