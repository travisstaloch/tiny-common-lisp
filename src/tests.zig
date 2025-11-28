const std = @import("std");
const testing = std.testing;
const t_gpa = testing.allocator;
const Interp = @import("root.zig");
const Expr = Interp.Expr;

fn expectExprEqual(expected: Expr.Fmt, actual: Expr.Fmt) !void {
    // trace("{f}\n{f}\n", .{ expected, actual });
    try testing.expectEqual(expected.e.boxed.tag, actual.e.boxed.tag);
    const ep = expected.p;
    const ap = actual.p;
    switch (expected.e.boxed.tag.int()) {
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
        else => try testing.expectEqual(expected.e.int, actual.e.int),
    }
}

const TestCase = union(enum) {
    file_path: []const u8,
    src: []const u8,
};

fn testParseFmt(c: TestCase) !void {
    var discarding = std.Io.Writer.Discarding.init(&.{});
    const N = 4096 * 4;
    var memory: [N]Expr = undefined;
    var l: Interp = .init(&memory, &discarding.writer, .quiet);
    const src = switch (c) {
        .file_path => try std.fs.cwd().readFileAllocOptions(t_gpa, c.file_path, 100000, null, .of(u8), 0),
        .src => try t_gpa.dupeZ(u8, c.src),
    };
    const file_path = switch (c) {
        .file_path => c.file_path,
        .src => "test",
    };
    defer t_gpa.free(src);
    const e = try l.parse(src, file_path);
    // trace("{s} {s}", .{ file_path, src });
    // trace("e {f}", .{e.fmt(&l)});
    const src2 = try std.fmt.allocPrintSentinel(t_gpa, "{f}", .{e.fmt(&l)}, 0);
    // trace("{s}", .{src2});
    defer t_gpa.free(src2);
    var memory2: [N]Expr = undefined;
    var l2: Interp = .init(&memory2, &discarding.writer, .quiet);
    const e2 = try l2.parse(src2, file_path);
    try expectExprEqual(e.fmt(&l), e2.fmt(&l2));
}

test "parse fmt rountrip" {
    try testParseFmt(.{ .file_path = "examples/basic.lisp" });
    try testParseFmt(.{ .file_path = "examples/fizzbuzz.lisp" });
    try testParseFmt(.{ .file_path = "examples/sqrt.lisp" });
    try testParseFmt(.{ .file_path = "tests/misc.lisp" });
}

pub fn testEval(expected: []const u8, src: [:0]const u8) !void {
    var discarding = std.Io.Writer.Discarding.init(&.{});
    const N = 4096 * 4;
    var memory: [N]Expr = undefined;
    var l: Interp = .init(&memory, &discarding.writer, .quiet);
    const e = try l.run(src, "<testEval>");
    try testing.expectFmt(expected, "{f}", .{e.fmt(&l)});
}

test "nil sym" {
    try testEval("()", "nil");
}
test "&rest" {
    try testEval("(1 2 (3 4 5))",
        \\(defun rest-as-list (a b &rest rest)
        \\  (list a b rest))
        \\(rest-as-list 1 2 3 4 5)
    );
}
