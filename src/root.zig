const std = @import("std");
const Type = std.builtin.Type;

pub const Tokenizer = @import("Tokenizer.zig");
pub const Parser = @import("Parser.zig");
pub const Ast = @import("Ast.zig");
pub const primitives = @import("primitives.zig");

test {
    _ = Tokenizer;
    _ = Parser;
    _ = Ast;
    _ = primitives;
}

// parse, format, parse round trip and check tokens are equal
fn checkFile(path: []const u8, gpa: std.mem.Allocator) !void {
    var arena_ = std.heap.ArenaAllocator.init(gpa);
    defer arena_.deinit();
    const arena = arena_.allocator();
    var f = try std.fs.cwd().openFile(path, .{});
    defer f.close();
    var fr = f.reader(&.{});
    const src1 = try arena.allocSentinel(u8, try f.getEndPos(), 0);
    std.debug.assert(src1.len == try fr.interface.readSliceShort(src1));

    const options: Ast.Options = .{ .mode = .{ .gpa = arena } };
    var p1 = try Parser.init(src1, options);
    var res1 = try p1.parse();
    var iter1 = res1.ast.root_lst.iterator(&res1.ast);
    const src2 = try std.fmt.allocPrintSentinel(arena, "{f}", .{iter1}, 0);
    var p2 = try Parser.init(src2, options);
    var res2 = try p2.parse();

    var iter2 = res2.ast.root_lst.iterator(&res2.ast);
    while (iter1.next()) |it1| {
        try std.testing.expectEqual(it1.id, iter2.next().?.id);
    }
    try std.testing.expect(iter2.next() == null);
}

const t_gpa = std.testing.allocator;
test "tokenize / format / tokenize" {
    try checkFile("examples/basic.scm", t_gpa);
    try checkFile("examples/fizzbuzz.scm", t_gpa);
}
