const std = @import("std");
const assert = std.debug.assert;
const deme = @import("deme");
const flagset = @import("flagset");

const flags = [_]flagset.Flag{
    .{ .type = []const u8, .name = "file", .options = .{ .kind = .positional } },
};

pub fn main() !void {
    var gpa_state = std.heap.GeneralPurposeAllocator(.{}){};
    defer assert(gpa_state.deinit() == .ok);
    const gpa = gpa_state.allocator();

    var args = flagset.parseFromIter(&flags, std.process.args(), .{ .allocator = gpa }) catch |e| switch (e) {
        error.HelpRequested => return,
        else => return e,
    };
    defer args.deinit(gpa);
    // std.debug.print("args {f}\n", .{flagset.fmtParsed(&flags, args.parsed, .{})});
    const f = try std.fs.cwd().openFile(args.parsed.file, .{});
    defer f.close();
    var read_buf: [256]u8 = undefined;
    var freader = f.reader(&read_buf);
    var w: std.Io.Writer.Allocating = try .initCapacity(gpa, try freader.getSize());
    defer w.deinit();
    try freader.interface.streamExact(&w.writer, w.writer.buffer.len);
    const end = w.writer.end;
    try w.writer.writeByte(0);
    const src = w.written()[0..end :0];
    var ast = try deme.Parser.parse(src, gpa);
    defer ast.deinit();
    std.debug.print("{f}\n{f}\n", .{ ast.root_env.iterator(&ast), ast.root_list.iterator(&ast) });
}

fn checkEval(s: []const u8) !void {
    _ = s; // autofix
    var ctx = deme.Ast.default_ast;
    defer ctx.deinit();
}

test {
    try checkEval(
        \\> (quote a)
        \\  a
        \\> 'a
        \\  a
        \\> (quote a b c)
        \\  (a b c)
        \\
    );
}
