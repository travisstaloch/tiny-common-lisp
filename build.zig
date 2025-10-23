const std = @import("std");
pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const mod = b.addModule("deme", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
    });
    const flagset = b.dependency("flagset", .{});
    const anyline = b.dependency("anyline", .{});
    const exe_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "deme", .module = mod },
            .{ .name = "flagset", .module = flagset.module("flagset") },
            .{ .name = "anyline", .module = anyline.module("anyline") },
        },
    });
    const exe = b.addExecutable(.{
        .name = "deme",
        .root_module = exe_mod,
        // .use_llvm = true,
    });
    b.installArtifact(exe);
    const run_step = b.step("run", "Run the app");
    const run_cmd = b.addRunArtifact(exe);
    run_step.dependOn(&run_cmd.step);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const mod_tests = b.addTest(.{ .root_module = mod });
    const run_mod_tests = b.addRunArtifact(mod_tests);
    const exe_tests = b.addTest(.{
        .root_module = exe.root_module,
        // .use_llvm = true,
        .filters = if (b.option([]const u8, "test-filter", "")) |f| b.dupeStrings(&.{f}) else &.{},
    });
    const run_exe_tests = b.addRunArtifact(exe_tests);
    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_mod_tests.step);
    test_step.dependOn(&run_exe_tests.step);

    const exe_check = b.addExecutable(.{ .name = "check", .root_module = exe_mod });
    const check = b.step("check", "Check if foo compiles");
    check.dependOn(&exe_check.step);
    check.dependOn(&exe_tests.step);
}
