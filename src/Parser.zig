const std = @import("std");
const assert = std.debug.assert;
const Parser = @This();

const Ast = @import("Ast.zig");
const Expr = Ast.Expr;
const Tokenizer = @import("Tokenizer.zig");
const primitives = @import("primitives.zig");

t: Tokenizer,
stack: std.ArrayList(StackItem) = .{},
ast: Ast,

const StackItem = struct { Expr.Id, Tokenizer.Token };

pub fn deinit(p: *Parser) void {
    p.exprs.deinit(p.gpa);
    p.stack.deinit(p.gpa);
    p.strs.deinit(p.gpa);
    p.root_env.deinit(p.gpa);
}

/// push 'e' to the cons at top of the stack.  then if e is a cons push to stack.
pub fn pushExpr(p: *Parser, e: Ast.Iterator, tok: Tokenizer.Token) !void {
    if (p.ast.options.mode == .gpa) {
        // std.debug.print("stack.len {}\n", .{p.stack.items.len});
        const lastid, _ = p.stack.items[p.stack.items.len - 1];
        // std.debug.print("pushExpr stack.len {} lastid {} {}\n", .{ p.stack.items.len, lastid, new.expr().* });
        _ = try lastid.iterator(&p.ast).append(e);
        if (e.expr().* == .cons) try p.stack.append(p.ast.options.mode.gpa, .{ e.id, tok });
    } else {
        const lastid: Expr.Id = @enumFromInt(0);
        _ = try lastid.iterator(&p.ast).append(e);
        p.stack.items.len += 1;
    }
}

pub const Error = std.mem.Allocator.Error || error{UnbalancedParens} || std.fmt.ParseFloatError;

fn stackAppend(p: *Parser, item: StackItem) !void {
    if (p.ast.options.mode == .gpa) {
        try p.stack.append(p.ast.options.mode.gpa, item);
    } else {
        p.stack.items.len += 1;
    }
}
fn tokAppend(p: *Parser, tok: Tokenizer.Token) !void {
    if (p.ast.options.mode == .gpa) {
        try p.ast.toks.append(p.ast.options.mode.gpa, tok);
    } else {
        p.ast.toks.items.len += 1;
    }
}

pub fn parse(src: [:0]const u8, options: Ast.Options) Error!Ast {
    var p: Parser = .{
        .t = .{ .src = src },
        .ast = .{ .options = options, .root_env = .nil, .root_lst = .nil },
    };

    p.ast.root_env = (try Ast.Env.init(&p.ast)).id;
    // std.debug.print("parse env {}\n", .{p.ast.getExpr(p.ast.root_env).env});

    defer if (options.mode == .gpa) p.stack.deinit(options.mode.gpa);
    errdefer p.ast.deinit();

    const root = try p.ast.createExpr(.{ .cons = .empty });
    try p.tokAppend(.empty);
    p.ast.root_lst = root.id;
    try p.stackAppend(.{ root.id, .{ .start = 0, .end = 0, .tag = .lparen } });
    defer {
        if (options.mode == .gpa) {
            const expected = p.ast.root_lst.iterator(&p.ast).expr().cons.fst.iterator(&p.ast).count();
            if (expected != p.ast.toks.items.len) {
                std.debug.print("expected {} tokens.  found {}\n{f}\n", .{ expected, p.ast.toks.items.len, p.ast.dump(src) });
            }
        }
    }

    while (true) {
        const tok = p.t.next();
        if (tok.tag != .rparen and tok.tag != .eof) try p.tokAppend(tok);

        // std.debug.print("{t: <10} {: >4}:{: >4} {s}\n", .{ tok.tag, tok.start, tok.end, tok.src(src) });
        switch (tok.tag) {
            .eof => break,
            .lparen => {
                const x = try p.ast.createExpr(.{ .cons = .empty });
                try p.pushExpr(x, tok);
            },
            .rparen => {
                if (options.mode == .gpa) {
                    _, const stok = p.stack.pop() orelse return error.UnbalancedParens;
                    if (stok.tag != .lparen) return error.UnbalancedParens;
                } else p.stack.items.len -= 1;
            },
            .string, .symbol => {
                const sym = try p.ast.internStr(tok.src(src));
                try p.pushExpr(sym.iterator(&p.ast), tok);
            },
            .number => {
                const num = try p.ast.createExpr(.{ .num = try std.fmt.parseFloat(f64, tok.src(src)) });
                try p.pushExpr(num, tok);
            },
            // .quasi_unquote => {
            //     // try p.ast.createExpr(.{ .lam = .{.arg_lst = .nil, .name = try p.ast.internStr() } });
            // },
            else => {
                std.debug.panic("TODO handle {t} '{s}' line '{s}'\n", .{ tok.tag, tok.src(src), tok.line(src) });
            },
        }
    }

    if (options.mode == .gpa and p.stack.items.len != 1) return error.UnbalancedParens;

    return p.ast;
}

const testing = std.testing;
const tgpa = testing.allocator;

test Parser {
    var ast = try Parser.parse("(+ 1 1)", .{ .mode = .{ .gpa = tgpa } });
    defer ast.deinit();

    { // check env
        var iter = ast.root_env.iterator(&ast).expr().env.scope_list.iterator(&ast);
        while (iter.next()) |sym| {
            try testing.expectEqual(.sym, sym.tag());
            try testing.expectEqual(.lam, iter.next().?.tag());
        }
    }

    { // check body
        var iter = ast.root_lst.iterator(&ast).expr().cons.fst.iterator(&ast);
        try testing.expectEqual(.sym, iter.next().?.tag());
        try testing.expectEqual(.num, iter.next().?.tag());
        try testing.expectEqual(.num, iter.next().?.tag());
        try testing.expectEqual(null, iter.next());
    }

    try testing.expectFmt("(+ 1 1)", "{f}", .{ast.root_lst.iterator(&ast)});
}

test "comptime parse" {
    var ast = try Parser.parse("(+ 1 1)", .{ .mode = .measure });
    defer ast.deinit();
}
