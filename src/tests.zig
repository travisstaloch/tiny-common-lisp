const std = @import("std");
const testing = std.testing;
const t_gpa = testing.allocator;
const Interp = @import("root.zig");
const Expr = Interp.Expr;

fn expectExprEqual(expected: Expr.Fmt, actual: Expr.Fmt) !void {
    // trace("{f}\n{f}\n", .{ expected, actual });
    const e = expected.e.expr(expected.p);
    const a = actual.e.expr(actual.p);
    try testing.expectEqual(e.boxed.tag, a.boxed.tag);
    const ep = expected.p;
    const ap = actual.p;
    switch (e.boxed.tag.int()) {
        Interp.ATOM => try testing.expectEqualStrings(
            ep.atomName(expected.e),
            ap.atomName(actual.e),
        ),
        Interp.CONS => {
            try expectExprEqual(
                ep.carAssume(expected.e).fmt(ep),
                ap.carAssume(actual.e).fmt(ap),
            );
            try expectExprEqual(
                ep.cdrAssume(expected.e).fmt(ep),
                ap.cdrAssume(actual.e).fmt(ap),
            );
        },
        else => try testing.expectEqual(e.int, a.int),
    }
}

const TestCase = union(enum) {
    file_path: []const u8,
    src: []const u8,
};

fn testParseFmt(c: TestCase) !void {
    var discarding = std.Io.Writer.Discarding.init(&.{});
    const N = 4096 * 4 * @sizeOf(Expr);
    var buffer: [N]u8 align(16) = undefined;
    var p: Interp = try .init(&buffer, &discarding.writer, .quiet);
    const src = switch (c) {
        .file_path => try std.fs.cwd().readFileAllocOptions(t_gpa, c.file_path, 100000, null, .of(u8), 0),
        .src => try t_gpa.dupeZ(u8, c.src),
    };
    const file_path = switch (c) {
        .file_path => c.file_path,
        .src => "test",
    };
    defer t_gpa.free(src);
    const e = try p.parse(src, file_path);
    // trace("{s} {s}", .{ file_path, src });
    // trace("e {f}", .{e.fmt(&l)});
    const src2 = try std.fmt.allocPrintSentinel(t_gpa, "{f}", .{e.fmt(&p)}, 0);
    // trace("{s}", .{src2});
    defer t_gpa.free(src2);
    var memory2: [N]u8 align(16) = undefined;
    var p2: Interp = try .init(&memory2, &discarding.writer, .quiet);
    const e2 = try p2.parse(src2, file_path);
    try expectExprEqual(e.fmt(&p), e2.fmt(&p2));
}

test "parse fmt rountrip" {
    try testParseFmt(.{ .file_path = "examples/basic.lisp" });
    try testParseFmt(.{ .file_path = "examples/fizzbuzz.lisp" });
    try testParseFmt(.{ .file_path = "examples/sqrt.lisp" });
    try testParseFmt(.{ .file_path = "tests/misc.lisp" });
    try testParseFmt(.{ .file_path = "tests/args.lisp" });
}

pub fn testEval(expected: []const u8, src: [:0]const u8) !void {
    var discarding = std.Io.Writer.Discarding.init(&.{});
    const N = 4096 * 4 * @sizeOf(Expr);
    var memory: [N]u8 align(16) = undefined;
    var p: Interp = try .init(&memory, &discarding.writer, .quiet);
    const e = try p.run(src, "<testEval>");
    try testing.expectFmt(expected, "{f}", .{e.fmt(&p)});
}

test "car cdr" {
    var discarding = std.Io.Writer.Discarding.init(&.{});
    const N = 4096 * 4 * @sizeOf(Expr);
    var memory: [N]u8 align(16) = undefined;
    var p: Interp = try .init(&memory, &discarding.writer, .quiet);
    const a = try p.atom("a");
    const b = try p.atom("b");
    const c = try p.cons(a, b);
    try testing.expectEqual(a.expr(&p).*, (try p.car(c)).expr(&p).*);
    try testing.expectEqual(b.expr(&p).*, (try p.cdr(c)).expr(&p).*);
    try testing.expectEqual(null, p.cdrOpt(b));
    try testing.expectEqual(null, p.cdrOpt(try p.cdr(c)));
}

test "nil sym" {
    try testEval("()", "nil");
}
