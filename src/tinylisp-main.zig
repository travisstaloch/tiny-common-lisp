const std = @import("std");
const flagset = @import("flagset");

const TinyLisp = @import("TinyLisp.zig");

pub const std_options: std.Options = .{
    // .log_level = .debug,
};

const flags = [_]flagset.Flag{
    .init(
        ?[]const u8,
        "script",
        .{
            .default_value_ptr = &@as(?[]const u8, null),
            .desc = "file path of script to run",
        },
    ),
};

pub fn main() !void {
    var alloc_buf: [4096 * 8]u8 = undefined;
    var fba = std.heap.FixedBufferAllocator.init(&alloc_buf);
    const gpa = fba.allocator();

    var stdout_buf: [256]u8 = undefined;
    var stdout_w = std.fs.File.stdout().writer(&stdout_buf);
    const stdout = &stdout_w.interface;
    defer stdout.flush() catch {};

    var args = flagset.parseFromIter(&flags, std.process.args(), .{ .allocator = gpa }) catch |e| switch (e) {
        error.HelpRequested => {
            try stdout.print("{f}", .{flagset.fmtUsage(&flags, ": <35", .full,
                \\
                \\usage: tinylisp <options>
                \\
                \\
            )});
            return;
        },
        else => return e,
    };
    defer args.deinit(gpa);

    try stdout.print("args: {f}\n", .{flagset.fmtParsed(&flags, args.parsed, .{})});

    if (args.parsed.script) |script| {
        const input_f = try std.fs.cwd().openFile(script, .{});
        defer input_f.close();
        try run(gpa, stdout, input_f, script, .script);
    } else {
        try run(gpa, stdout, std.fs.File.stdin(), "<stdin>", .repl);
    }
}

fn run(
    gpa: std.mem.Allocator,
    output: *std.Io.Writer,
    input_f: std.fs.File,
    name: []const u8,
    mode: enum { script, repl },
) !void {
    try output.writeAll(
        // banner genrated by
        // https://patorjk.com/software/taag/#p=display&f=Small+Block&t=TinyLisp&x=none&v=4&h=4&w=80&we=false
        \\
        \\  ▀▛▘▗       ▌  ▗       
        \\   ▌ ▄ ▛▀▖▌ ▌▌  ▄ ▞▀▘▛▀▖
        \\   ▌ ▐ ▌ ▌▚▄▌▌  ▐ ▝▀▖▙▄▘
        \\   ▘ ▀▘▘ ▘▗▄▘▀▀▘▀▘▀▀ ▌  
        \\    TinyLisp 0.0.1
        \\  
        \\
    );
    try output.flush();

    var wa: std.Io.Writer.Allocating = .init(gpa);
    var memory: [1024 * 16]TinyLisp.Expr = undefined;
    var p: TinyLisp = .init(&memory, output);

    var input_buf: [4096]u8 = undefined;

    switch (mode) {
        .repl => while (true) {
            try output.print("{: >4}> ", .{p.sp - p.hp / 8});
            try output.flush();
            wa.clearRetainingCapacity();

            while (true) {
                const amt = try input_f.read(&input_buf);
                if (amt == 0 or (amt == 1 and input_buf[0] == '\n')) break;
                try wa.writer.writeAll(input_buf[0..amt]);
            }
            try wa.writer.writeByte(0);
            const src = wa.written()[0 .. wa.written().len - 1 :0];
            if (src.len <= 1) break;

            _ = try p.run(src, name);
            p.gc();
        },
        .script => {
            var input = input_f.reader(&input_buf);
            _ = try input.interface.stream(&wa.writer, .unlimited);
            try wa.writer.writeByte(0);
            const src = wa.written()[0 .. wa.written().len - 1 :0];
            const e = try p.run(src, name);
            try output.print("\n{: >4}> {f}\n", .{ p.sp - p.hp / 8, e.fmt(&p) });
            try output.flush();
            // try output.print("{f}\n", .{e.fmt(&l)});
        },
    }
}
