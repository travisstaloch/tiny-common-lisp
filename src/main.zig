const std = @import("std");
const assert = std.debug.assert;
const deme = @import("deme");
const flagset = @import("flagset");
const anyline = @import("anyline");

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
    var p = try deme.Parser.init(src, .{ .mode = .{ .gpa = gpa } });
    var res = try p.parse();
    defer res.ast.deinit();
    std.debug.print("{f}\n{f}\n{f}\n", .{ res.ast.root_env.iterator(&res.ast), res.ast.root_lst.iterator(&res.ast), res.ast.dump(src) });
    anyline.using_history();

    var arena = std.heap.ArenaAllocator.init(gpa);
    while (true) {
        const line = try anyline.readline(arena.allocator(), " > ");
        std.debug.print("{s}", .{line});
        try anyline.add_history(gpa, line);
    }
}
