const std = @import("std");
const flagset = @import("flagset");

const TinyLisp = @import("root.zig");

pub const std_options: std.Options = .{
    // .log_level = .debug,
};

const flags = [_]flagset.Flag{
    .init([]const u8, "script", .{
        .short = 's',
        .desc = "file path of script to run",
        .kind = .list,
    }),
    .init([]const u8, "eval", .{ .short = 'e', .desc = "text to eval", .kind = .list }),
    .init([]const u8, "load", .{ .short = 'l', .desc = "file to load", .kind = .list }),
    .init(bool, "banner", .{ .desc = "show banner.  default true.", .default_value_ptr = &true }),
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
                \\usage: tiny-common-lisp <options>
                \\
                \\
            )});
            try stdout.flush();
            return;
        },
        else => return e,
    };
    defer args.deinit(gpa);

    if (args.parsed.banner) {
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
    }

    var memory: [1024 * 16]TinyLisp.Expr = undefined;
    var p: TinyLisp = .init(&memory, stdout, .quiet);

    if (args.parsed.eval.items.len > 0) {
        for (args.parsed.eval.items) |eval| {
            try run(gpa, &p, "<eval>", .{ .eval = eval });
        }
    }
    if (args.parsed.load.items.len > 0) {
        unreachable; // TODO
    }
    if (args.parsed.script.items.len > 0) {
        for (args.parsed.script.items) |script| {
            const input_f = std.fs.cwd().openFile(script, .{}) catch |e| {
                std.debug.print("failed to open file: {s}: error.{t}\n", .{ script, e });
                break;
            };
            defer input_f.close();
            try run(gpa, &p, script, .{ .script = input_f });
        }
    } else {
        try run(gpa, &p, "<stdin>", .{ .repl = std.fs.File.stdin() });
    }
}

fn run(
    gpa: std.mem.Allocator,
    p: *TinyLisp,
    name: []const u8,
    mode: union(enum) {
        script: std.fs.File,
        repl: std.fs.File,
        eval: []const u8,
    },
) !void {
    var wa: std.Io.Writer.Allocating = .init(gpa);
    var input_buf: [4096]u8 = undefined;

    switch (mode) {
        .repl => |input_f| while (true) {
            p.print_mode = .repl;
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
            p.gcOld();
        },
        .script => |input_f| {
            var input = input_f.reader(&input_buf);
            _ = try input.interface.stream(&wa.writer, .unlimited);
            try wa.writer.writeByte(0);
            const src = wa.written()[0 .. wa.written().len - 1 :0];
            _ = try p.run(src, name);
        },
        .eval => |to_eval| {
            try wa.writer.writeAll(to_eval);
            try wa.writer.writeByte(0);
            const src = wa.written()[0 .. wa.written().len - 1 :0];
            _ = try p.run(src, name);
        },
    }
}
