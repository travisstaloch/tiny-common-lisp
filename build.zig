const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const build_opts = b.addOptions();
    build_opts.addOption(bool, "trace", b.option(bool, "trace", "") orelse false);

    const RegionAllocator = b.dependency("RegionAllocator", .{});
    const mod = b.addModule("tiny-common-lisp", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .imports = &.{
            .{ .name = "build-options", .module = build_opts.createModule() },
            .{ .name = "Region", .module = RegionAllocator.module("RegionAllocator") },
        },
    });
    const flagset = b.dependency("flagset", .{});
    const exe_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "flagset", .module = flagset.module("flagset") },
            .{ .name = "Region", .module = RegionAllocator.module("RegionAllocator") },
            .{ .name = "build-options", .module = build_opts.createModule() },
        },
    });
    const exe = b.addExecutable(.{
        .name = "tiny-common-lisp",
        .root_module = exe_mod,
        .use_llvm = true,
    });
    b.installArtifact(exe);
    const run_step = b.step("run", "Run the app");
    const run_cmd = b.addRunArtifact(exe);
    run_step.dependOn(&run_cmd.step);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_cmd.addArgs(args);

    const filters_opt = b.option([]const u8, "test-filter", "");
    const filters = if (filters_opt) |f| b.dupeStrings(&.{f}) else &.{};
    const mod_tests = b.addTest(.{ .root_module = mod, .filters = filters });
    const run_mod_tests = b.addRunArtifact(mod_tests);
    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_mod_tests.step);
    b.installArtifact(mod_tests);

    const exe_check = b.addExecutable(.{ .name = "check", .root_module = exe_mod });
    const check = b.step("check", "Check if everything compiles");
    check.dependOn(&exe_check.step);
    check.dependOn(&mod_tests.step);
}
