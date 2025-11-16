const std = @import("std");

const TinyLisp = @import("TinyLisp.zig");

pub const std_options: std.Options = .{
    .log_level = .debug,
};

pub fn main() !void {
    var alloc_buf: [4096]u8 = undefined;
    var fba = std.heap.FixedBufferAllocator.init(&alloc_buf);
    const gpa = fba.allocator();

    var stdout_buf: [256]u8 = undefined;
    var stdout_w = std.fs.File.stdout().writer(&stdout_buf);
    const stdout = &stdout_w.interface;
    defer stdout.flush() catch {};

    const args = try std.process.argsAlloc(gpa);

    var input_buf: [4096]u8 = undefined;
    var input_f: std.fs.File, const name = if (args.len > 1)
        .{ try std.fs.cwd().openFile(args[1], .{}), args[1] }
    else
        .{ std.fs.File.stdin(), "<stdin>" };
    defer input_f.close();

    var input = input_f.reader(&input_buf);
    var wa: std.Io.Writer.Allocating = .init(gpa);
    _ = try input.interface.stream(&wa.writer, .unlimited);
    try wa.writer.writeByte(0);
    const src = wa.written()[0 .. wa.written().len - 1 :0];
    // std.debug.print("src {s}\n", .{src});

    var memory: [1024]TinyLisp.Expr = undefined;

    var l: TinyLisp = .init(&memory, stdout);

    try stdout.writeAll("tinylisp\n");
    try stdout.print("\n{: >4}> ", .{l.sp - l.hp / 8});
    try stdout.flush();
    const e = try l.run(src, name);
    try stdout.print("{f}\n", .{e.fmt(&l)});
    l.gc();
}
