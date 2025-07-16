const std = @import("std");

// Although this function looks imperative, note that its job is to
// declaratively construct a build graph that will be executed by an external
// runner.
pub fn build(b: *std.Build) void {
    // Standard target options allows the person running `zig build` to choose
    // what target to build for. Here we do not override the defaults, which
    // means any target is allowed, and the default is native. Other options
    // for restricting supported target set are available.
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const dvui_dep = b.dependency("dvui", .{ .backend = .sdl3, .target = target, .optimize = optimize });

    const lib = b.addStaticLibrary(.{
        .name = "s3-vk",
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    b.installArtifact(lib);

    const exe = b.addExecutable(.{
        .name = "s3-vk",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    //exe.want_lto = false;
    b.installArtifact(exe);

    // Run
    const run_cmd = b.addRunArtifact(exe);
    // By making the run step depend on the install step, it will be run from the
    // installation directory rather than directly from within the cache directory.
    run_cmd.step.dependOn(b.getInstallStep());
    // args from build: -- arg1 arg2 etc
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }
    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    // Tests
    const lib_unit_tests = b.addTest(.{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    const run_lib_unit_tests = b.addRunArtifact(lib_unit_tests);
    const exe_unit_tests = b.addTest(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    const run_exe_unit_tests = b.addRunArtifact(exe_unit_tests);
    // zig build test
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_lib_unit_tests.step);
    test_step.dependOn(&run_exe_unit_tests.step);

    // dvui
    // exe_unit_tests.root_module.addImport("dvui", dvui_dep.module("dvui_sdl"));
    exe.root_module.addImport("dvui", dvui_dep.module("dvui_sdl3"));

    // SDL
    exe.addIncludePath(b.path("deps/SDL/include"));
    exe_unit_tests.addIncludePath(b.path("deps/SDL/include"));

    // Vulkan
    const vk_registry_opt = b.option([]const u8, "vk_registry", "Path to vulkan registry vk.xml");
    const vk_registry = vk_registry_opt orelse blk: {
        const env = std.process.getEnvMap(b.allocator) catch unreachable;
        if (env.get("VULKAN_SDK")) |vk_path| {
            break :blk b.pathJoin(&.{ vk_path, "share", "vulkan", "registry", "vk.xml" });
        }
        std.log.err("VULKAN_SDK not found. Pass in -Dvk_registry=/path/to/vk.xml or install vulkan SDK.", .{});
        break :blk "/usr/share/vulkan/registry/vk.xml"; // best guess
    };
    const vkzig_dep = b.dependency("vulkan_zig", .{
        .registry = @as([]const u8, vk_registry),
    });
    const vkzig_bindings = vkzig_dep.module("vulkan-zig");
    exe.root_module.addImport("vulkan", vkzig_bindings);
    exe_unit_tests.root_module.addImport("vulkan", vkzig_bindings);

    { // Shaders
        const glslc = b.option(bool, "glslc", "Compile glsl shaders") orelse false;
        const slangc = b.option(bool, "slangc", "Compile slang shaders") orelse false;
        _ = slangc; // autofix

        const shader_subpath = "shaders";
        const dir = std.fs.cwd().openDir(shader_subpath, .{ .iterate = true }) catch unreachable;
        var it = dir.iterate();
        while (it.next() catch unreachable) |f| {
            if (f.kind == .file) {
                const is_glsl =
                    std.mem.endsWith(u8, f.name, ".vert") or
                    std.mem.endsWith(u8, f.name, ".frag") or
                    std.mem.endsWith(u8, f.name, ".tesc") or
                    std.mem.endsWith(u8, f.name, ".tese") or
                    std.mem.endsWith(u8, f.name, ".geom") or
                    std.mem.endsWith(u8, f.name, ".comp");
                const is_slang = std.mem.endsWith(u8, f.name, ".slang");

                if (is_slang and !glslc) {
                    const shader_path = b.pathJoin(&.{ shader_subpath, f.name });
                    const ShaderType = enum { vertex, fragment, compute };
                    _ = ShaderType; // autofix
                    const file_contents = std.fs.cwd().readFileAlloc(b.allocator, shader_path, 10 * 1024 * 1024) catch unreachable;
                    defer b.allocator.free(file_contents);
                    const shader_types: []const []const u8 = &.{ "[shader(\"vertex\")]", "[shader(\"fragment\")]" };
                    for (shader_types, 0..) |_, shader_type_idx| {
                        if (std.mem.indexOfAnyPos(u8, file_contents, 0, shader_types[shader_type_idx])) |_| {
                            const out_name = std.mem.join(b.allocator, "", &.{
                                f.name[0 .. f.name.len - ".slang".len],
                                switch (shader_type_idx) {
                                    0 => ".vert",
                                    1 => ".frag",
                                    else => unreachable,
                                },
                                ".spv",
                            }) catch unreachable;
                            const out_path = b.pathJoin(&.{ shader_subpath, out_name });
                            const compile = b.addSystemCommand(&.{
                                "slangc",
                                "-target",
                                "spirv",
                                "-entry",
                                switch (shader_type_idx) {
                                    0 => "vertexMain",
                                    1 => "fragmentMain",
                                    else => unreachable,
                                },
                                "-o",
                            });
                            compile.addArg(out_name); // output file
                            if (optimize == .Debug) compile.addArg("-minimum-slang-optimization") else compile.addArg("-O3");
                            compile.addArg(f.name); // input file

                            compile.setCwd(b.path(shader_subpath));

                            const gf = b.allocator.create(std.Build.GeneratedFile) catch unreachable;
                            gf.* = std.Build.GeneratedFile{ .step = &compile.step, .path = out_path };
                            const out = std.Build.LazyPath{ .generated = .{ .file = gf } };
                            exe.root_module.addAnonymousImport(out_name, .{
                                .root_source_file = out,
                            });
                            exe_unit_tests.root_module.addAnonymousImport(out_name, .{
                                .root_source_file = out,
                            });
                            exe.step.dependOn(&compile.step);
                            //exe.step.dependOn(&b.addInstallFile(out, b.pathJoin(&.{ "shaders", out_name })).step);
                        }
                    }
                }

                if (is_glsl) {
                    const out_name = std.mem.join(b.allocator, "", &.{ f.name, ".spv" }) catch unreachable;
                    const out_path = b.pathJoin(&.{ shader_subpath, out_name });
                    const out = blk: {
                        if (!glslc) break :blk b.path(out_path); // compilation not requested, just point to output

                        const compile = b.addSystemCommand(&.{
                            "glslc",
                            "--target-env=vulkan1.2",
                            "-o",
                        });
                        compile.setCwd(b.path(shader_subpath));
                        compile.addArg(out_name); // output file
                        compile.addArg(f.name); // input file

                        const gf = b.allocator.create(std.Build.GeneratedFile) catch unreachable;
                        gf.* = std.Build.GeneratedFile{ .step = &compile.step, .path = out_path };
                        break :blk std.Build.LazyPath{ .generated = .{ .file = gf } };
                    };
                    exe.root_module.addAnonymousImport(out_name, .{
                        .root_source_file = out,
                    });
                    exe_unit_tests.root_module.addAnonymousImport(out_name, .{
                        .root_source_file = out,
                    });
                    //exe.step.dependOn(&b.addInstallFile(out, b.pathJoin(&.{ "shaders", out_name })).step);
                }
            }
        }
    }
}
