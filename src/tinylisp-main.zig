const std = @import("std");
const flagset = @import("flagset");

const TinyLisp = @import("TinyLisp.zig");

pub const std_options: std.Options = .{
    // .log_level = .debug,
};

const flags = [_]flagset.Flag{
    .init([]const u8, "script", .{
        .short = 's',
        .desc = "file path of script to run",
        .kind = .list,
    }),
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

    try stdout.print(
        // banner genrated at
        // https://patorjk.com/software/taag/#p=display&f=Small+Block&t=TinyLisp&x=none&v=4&h=4&w=80&we=false
        \\
        \\  ▀▛▘▗       ▌  ▗       
        \\   ▌ ▄ ▛▀▖▌ ▌▌  ▄ ▞▀▘▛▀▖
        \\   ▌ ▐ ▌ ▌▚▄▌▌  ▐ ▝▀▖▙▄▘
        \\   ▘ ▀▘▘ ▘▗▄▘▀▀▘▀▘▀▀ ▌  
        \\    TinyLisp 0.0.1
        \\
        \\args: {f}
        \\
        \\
    , .{flagset.fmtParsed(&flags, args.parsed, .{})});
    try stdout.flush();

    var memory: [1024 * 16]TinyLisp.Expr = undefined;
    var p: TinyLisp = .init(&memory, stdout);

    if (args.parsed.script.items.len > 0) {
        for (args.parsed.script.items) |script| {
            const input_f = std.fs.cwd().openFile(script, .{}) catch |e| {
                std.debug.print("failed to open file: {s}: error.{t}\n", .{ script, e });
                break;
            };
            defer input_f.close();
            try run(gpa, &p, input_f, script, .script);
        }
    } else {
        try run(gpa, &p, std.fs.File.stdin(), "<stdin>", .repl);
    }
}

fn run(
    gpa: std.mem.Allocator,
    p: *TinyLisp,
    input_f: std.fs.File,
    name: []const u8,
    mode: enum { script, repl },
) !void {
    var wa: std.Io.Writer.Allocating = .init(gpa);

    var input_buf: [4096]u8 = undefined;

    switch (mode) {
        .repl => while (true) {
            try p.w.print("{: >4}> ", .{p.sp - p.hp / 8});
            try p.w.flush();
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
            _ = try p.run(src, name);
            // try p.w.print("\n{: >4}> {f}\n", .{ p.sp - p.hp / 8, e.fmt(p) });
            // try p.w.flush();
        },
    }
}
