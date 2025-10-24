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

/// push 'e' to the cons at top of the stack.  then if e is a cons push to stack.
pub fn pushExpr(p: *Parser, e: Ast.Iterator, tok: Tokenizer.Token) !void {
    switch (p.ast.options.mode) {
        .gpa => |gpa| {
            const lastid, _ = p.stack.items[p.stack.items.len - 1];
            _ = try lastid.iterator(&p.ast).append(e);
            if (e.expr().* == .cons) try p.stack.append(gpa, .{ e.id, tok });
        },
        .bounded => {
            const lastid, _ = p.stack.items[p.stack.items.len - 1];
            _ = try lastid.iterator(&p.ast).append(e);
            if (e.expr().* == .cons) p.stack.appendAssumeCapacity(.{ e.id, tok });
        },
        .measure => {
            const lastid: Expr.Id = @enumFromInt(0);
            _ = try lastid.iterator(&p.ast).append(e);
            p.stack.items.len += 1;
            p.stack.capacity = @max(p.stack.capacity, p.stack.items.len);
        },
    }
}

pub const Error = std.mem.Allocator.Error || error{UnbalancedParens} || std.fmt.ParseFloatError;

fn stackAppend(p: *Parser, item: StackItem) !void {
    switch (p.ast.options.mode) {
        .gpa => |gpa| try p.stack.append(gpa, item),
        .measure => {
            p.stack.items.len += 1;
            p.stack.capacity = @max(p.stack.capacity, p.stack.items.len);
        },
        .bounded => p.stack.appendAssumeCapacity(item),
    }
}

fn tokAppend(p: *Parser, tok: Tokenizer.Token) !void {
    if (true) return;
    switch (p.ast.options.mode) {
        .gpa => |gpa| try p.ast.toks.append(gpa, tok),
        .bounded => p.ast.toks.appendAssumeCapacity(tok),
        .measure => p.ast.toks.items.len += 1,
    }
}

/// returns an empty parser.  returned ast has given options and initialized root_env.
pub fn init(src: [:0]const u8, options: Ast.Options) Error!Parser {
    var p: Parser = .{
        .t = .{ .src = src },
        .ast = .{ .options = options, .root_env = .nil, .root_lst = .nil },
    };

    p.ast.root_env = (try Ast.Env.init(&p.ast)).id;
    return p;
}

const ParseResult = struct { ast: Ast, stack_capacity: usize };

pub fn parse(p: *Parser) Error!ParseResult {
    // std.debug.print("parse env {}\n", .{p.ast.getExpr(p.ast.root_env).env});
    defer switch (p.ast.options.mode) {
        .gpa => |gpa| p.stack.deinit(gpa),
        .bounded, .measure => {},
    };
    errdefer p.ast.deinit();

    const root = try p.ast.createExpr(.{ .cons = .empty });
    try p.tokAppend(.empty);
    p.ast.root_lst = root.id;
    try p.stackAppend(.{ root.id, .{ .start = 0, .end = 0, .tag = .lparen } });
    defer {
        if (p.ast.options.mode == .gpa) {
            // const expected = p.ast.root_lst.iterator(&p.ast).expr().cons.fst.iterator(&p.ast).count();
            // if (expected != p.ast.toks.items.len) {
            //     std.debug.print("expected {} tokens.  found {}\n{f}\n", .{ expected, p.ast.toks.items.len, p.ast.dump(src) });
            // }
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
                switch (p.ast.options.mode) {
                    .gpa, .bounded => {
                        _, const stok = p.stack.pop() orelse return error.UnbalancedParens;
                        if (stok.tag != .lparen) return error.UnbalancedParens;
                    },
                    .measure => p.stack.items.len -= 1,
                }
            },
            .string, .symbol => {
                const sym = try p.ast.internStr(tok.src(p.t.src));
                try p.pushExpr(sym.iterator(&p.ast), tok);
            },
            .number => {
                const num = try p.ast.createExpr(.{ .num = try std.fmt.parseFloat(f64, tok.src(p.t.src)) });
                try p.pushExpr(num, tok);
            },
            // .quasi_unquote => {
            //     // try p.ast.createExpr(.{ .lam = .{.arg_lst = .nil, .name = try p.ast.internStr() } });
            // },
            else => {
                std.debug.panic("TODO handle {t} '{s}' line '{s}'\n", .{ tok.tag, tok.src(p.t.src), tok.line(p.t.src) });
            },
        }
    }

    switch (p.ast.options.mode) {
        .gpa, .bounded => {
            if (p.stack.items.len != 1) return error.UnbalancedParens;
        },
        .measure => {},
    }

    return .{ .ast = p.ast, .stack_capacity = p.stack.capacity };
}

/// parse into the given buffers.
pub fn parseBounded(src: [:0]const u8, exprs: []Expr, syms: []Ast.SymItem, stack: []StackItem) Error!ParseResult {
    const ast: Ast = .{
        .exprs = .initBuffer(exprs),
        .syms = .initBuffer(syms),
        .options = .{ .mode = .bounded },
        .root_env = .nil,
        .root_lst = .nil,
    };
    var p: Parser = .{ .ast = ast, .stack = .initBuffer(stack), .t = .{ .src = src } };
    return try p.parse();
}

pub inline fn parseComptime(comptime src: [:0]const u8) Error!ParseResult {
    comptime {
        var p = try Parser.init(src, .{ .mode = .measure });
        const measured_res = try p.parse();

        var exprs: [measured_res.ast.exprs.items.len]Expr = undefined;
        var syms: [measured_res.ast.syms.items.len]Ast.SymItem = undefined;
        var stack: [measured_res.stack_capacity]StackItem = undefined;
        const res = try Parser.parseBounded(src, &exprs, &syms, &stack);

        const fexprs = exprs[0..].*;
        const fsyms = syms[0..].*;
        var ret: ParseResult = .{
            .stack_capacity = measured_res.stack_capacity,
            .ast = .{
                .exprs = .initBuffer(@constCast(&fexprs)), // FIXME @constCast
                .syms = .initBuffer(@constCast(&fsyms)),
                .root_env = res.ast.root_env,
                .root_lst = res.ast.root_lst,
                .options = res.ast.options,
            },
        };
        ret.ast.exprs.expandToCapacity();
        ret.ast.syms.expandToCapacity();
        return ret;
    }
}

const testing = std.testing;
const t_gpa = testing.allocator;

test Parser {
    var p = try Parser.init("(+ 1 1)", .{ .mode = .{ .gpa = t_gpa } });
    var res = try p.parse();
    const ast = &res.ast;
    defer ast.deinit();

    { // check env
        var iter = ast.root_env.iterator(ast).expr().env.scope_list.iterator(ast);
        while (iter.next()) |sym| {
            try testing.expectEqual(.sym, sym.tag());
            try testing.expectEqual(.lam, iter.next().?.tag());
        }
    }

    { // check body
        var iter = ast.root_lst.iterator(ast).expr().cons.fst.iterator(ast);
        try testing.expectEqual(.sym, iter.next().?.tag());
        try testing.expectEqual(.num, iter.next().?.tag());
        try testing.expectEqual(.num, iter.next().?.tag());
        try testing.expectEqual(null, iter.next());
    }

    try testing.expectFmt("(+ 1 1)", "{f}", .{ast.root_lst.iterator(ast)});
}

test "comptime parse" {
    const src = "(+ 1 1)";
    var res = try Parser.parseComptime(src);
    try testing.expectFmt(src, "{f}", .{res.ast.root_lst.iterator(&res.ast)});
}
