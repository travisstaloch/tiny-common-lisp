const std = @import("std");
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;
const Tokenizer = @import("Tokenizer.zig");
const Token = Tokenizer.Token;
const primitives = @import("primitives.zig");
const t_gpa = std.testing.allocator;
pub const default_ast = Ast{ .gpa = t_gpa, .root_env = .nil, .root_list = undefined };
const Parser = @import("Parser.zig");
pub const Error = error{Eval} || Allocator.Error || Parser.Error;

const Ast = @This();

gpa: std.mem.Allocator,
exprs: std.ArrayList(Expr) = .{},
syms: std.StringArrayHashMapUnmanaged(Expr.Id) = .{},
root_env: Expr.Id,
root_list: Expr.Id,

pub fn deinit(ast: *Ast) void {
    ast.exprs.deinit(ast.gpa);
    ast.syms.deinit(ast.gpa);
}

pub fn createExpr(ast: *Ast, e: Expr) !Ast.Iterator {
    const id: Expr.Id = @enumFromInt(ast.exprs.items.len);
    try ast.exprs.append(ast.gpa, e);
    return .init(id, ast);
}

pub fn internStr(ast: *Ast, str: []const u8) !Expr.Id {
    const sym: Ast.Sym = @enumFromInt(ast.syms.count());
    const gop = try ast.syms.getOrPut(ast.gpa, str);
    if (gop.found_existing) return gop.value_ptr.*;
    const e = try ast.createExpr(.{ .sym = sym });
    gop.value_ptr.* = e.id;
    return e.id;
}

pub fn expr(ast: *Ast, id: Expr.Id) *Expr {
    return &ast.exprs.items[@intFromEnum(id)];
}

pub fn newList(ast: *Ast, exprs: []const Expr) !Ast.Iterator {
    // var head = try ast.newExpr(.empty_lst);
    // if (exprs.len == 0) return head;
    // var i = exprs.len;
    // while (i > 0) {
    //     i -= 1;
    //     const e = try ast.newExpr(exprs[i]);
    //     head = try head.prepend(e);
    // }
    // return head;
    var head = try ast.createExpr(.empty_lst);
    for (exprs) |e| {
        const ee = try ast.createExpr(e);
        _ = try head.append(ee);
    }
    return head;
}

pub fn debug(ast: *Ast, w: *std.Io.Writer) !void {
    for (ast.exprs.items, 0..) |e, i| {
        try w.print("{: >4} {}\n", .{ i, e });
    }
}

pub const Env = struct {
    parent: Expr.Id,
    /// a list of key value pairs
    scope_list: Expr.Id,

    pub fn init(ast: *Ast) !Ast.Iterator {
        const root_scope = try ast.createExpr(.{ .cons = .empty });
        const root_env = try ast.createExpr(.{ .env = .{ .parent = .nil, .scope_list = root_scope.id } });
        // std.debug.print("root_env {f}\n", .{root_env});

        for (primitives.tags) |prim| {
            const sym = try ast.internStr(@tagName(prim));
            _ = try root_scope.append(sym.iterator(ast));
            const lam = try ast.createExpr(.{ .lam = .{
                .arg_lst = .nil,
                .name = if (is_debug) sym else {},
                .func = primitives.getFunc(prim),
                .env = root_env.id,
            } });
            _ = try root_scope.append(lam);
        }

        return root_env;
    }
};

const is_debug = @import("builtin").mode == .Debug;

pub const Lam = struct {
    /// a list of symbols
    arg_lst: Ast.Expr.Id,
    env: Ast.Expr.Id,

    name: if (is_debug) Expr.Id else void,
    func: primitives.Func,
};

pub const Sym = enum(usize) {
    _,

    pub fn str(s: Sym, ast: *Ast) []const u8 {
        return ast.syms.keys()[@intFromEnum(s)];
    }
};

pub const Expr = union(enum) {
    sym: Sym,
    num: f64,
    cons: Cons, // TODO cache last so append() doesn't have to walk to end
    lam: Lam,
    env: Env,

    pub const Tag = std.meta.Tag(Expr);
    pub const empty_lst: Expr = .{ .cons = .empty };

    pub fn getId(e: *const Expr, ast: *Ast) Id {
        // TODO better way to catch stack pointers.  this assumes 8MB stack.
        assert(e - ast.exprs.items.ptr < 1024 * 1024 * 8 / @sizeOf(Expr));
        return @enumFromInt(e - ast.exprs.items.ptr);
    }

    pub const Id = enum(usize) {
        /// aka empty list
        nil = std.math.maxInt(usize),
        _,

        pub const iterator = Iterator.init;
    };

    pub fn builder(e: *const Expr, ast: *Ast) Iterator {
        return .init(e.getId(ast), ast);
    }
};

/// methods for Expr.Ids
pub const Iterator = struct {
    id: Expr.Id,
    ast: *Ast,

    pub fn init(id: Expr.Id, ast: *Ast) Iterator {
        return .{ .id = id, .ast = ast };
    }

    pub fn expr(i: Iterator) *Expr {
        return &i.ast.exprs.items[@intFromEnum(i.id)];
    }

    pub fn as(i: Iterator, comptime t: Expr.Tag) @FieldType(Expr, @tagName(t)) {
        return @field(i.expr(), @tagName(t));
    }

    pub fn tag(i: Iterator) Expr.Tag {
        return i.expr().*;
    }

    /// prepend e to lst.  returns new head.
    /// assumes lst is a cons.
    pub fn prepend(lst: Iterator, elem: Iterator) !Iterator {
        const lst_ptr = lst.expr();
        const fst = lst_ptr.cons.fst;
        if (fst == .nil) {
            lst_ptr.cons.fst = elem.id;
            return lst;
        }

        const new = try lst.ast.createExpr(.{ .cons = .{ .fst = elem.id, .snd = fst } });
        lst.expr().cons.fst = new.id;
        return new;
    }

    /// assumes lst is a cons.
    pub fn append(lst: Iterator, elem: Iterator) !Iterator {
        // [o|o] -- [o|/]
        //  1        2

        const ex = lst.expr();
        const c = ex.cons;
        if (c.fst == .nil) {
            ex.cons.fst = elem.id;
            return lst;
        } else if (c.snd == .nil) {
            const snd = try lst.ast.createExpr(.{ .cons = .{ .fst = elem.id, .snd = .nil } });
            lst.expr().cons.snd = snd.id;
            return lst;
        } else {
            // FIXME always tail call once self hosted x86 supports it
            // @import("builtin").zig_backend == .stage2_x86_64
            // return try @call(.always_tail, append, .{ l.snd.b(lst.ast), e });
            return try c.snd.iterator(lst.ast).append(elem);
        }
    }

    pub fn format(i: Iterator, w: *std.Io.Writer) !void {
        // std.debug.print("format id {}\n", .{b.id});
        if (i.id == .nil) return;
        const e: [*]Expr = @ptrCast(i.expr());
        // try w.print("{t}-{}\n", .{ e[0], b.id });
        switch (e[0]) {
            .sym => |s| _ = try w.write(s.str(i.ast)),
            .num => |s| _ = try w.print("{d}", .{s}),
            .cons => {
                if (i.id != i.ast.root_list) _ = try w.write("("); // skip paren at root
                // try w.print("{} . {}\n", .{ l.fst, l.snd });
                var idx: usize = 0;
                var iter = i;
                while (iter.next()) |id| : (idx += 1) {
                    if (idx != 0) _ = try w.write(" ");
                    try id.format(w);
                }
                if (i.id != i.ast.root_list) _ = try w.write(")"); // skip paren at root
            },
            .env => |x| {
                try w.print("(env {f})", .{x.scope_list.iterator(i.ast)});
            },
            .lam => |x| {
                try w.print("'{s}", .{x.name.iterator(i.ast).as(.sym).str(i.ast)});
            },

            // else => |x| std.debug.panic("TODO {t}", .{x}),
        }
        try w.flush();
    }

    pub fn next(i: *Iterator) ?Iterator {
        if (i.id == .nil) return null;
        const cons = i.expr().cons;
        if (cons.fst == .nil) return null;

        i.id = if (cons.snd == .nil)
            .nil
        else
            cons.snd;

        return .{ .id = cons.fst, .ast = i.ast };
    }
};

pub const Cons = struct {
    fst: Expr.Id,
    snd: Expr.Id,

    pub const empty: Cons = .{ .fst = .nil, .snd = .nil };

    pub fn isEmpty(c: Cons) bool {
        return c.fst == .nil;
    }
};

test Cons {
    var ast = default_ast;
    defer ast.deinit();
    var iter = try ast.newList(&.{ .{ .num = 1 }, .{ .num = 2 } });

    try std.testing.expectEqual(1, iter.next().?.expr().num);
    try std.testing.expectEqual(2, iter.next().?.expr().num);
    try std.testing.expectEqual(null, iter.next());
}

test {
    _ = Ast.Cons;
    _ =
        \\
        \\10
        \\(+ 5 3 4)
        \\(- 9 1)
        \\(/ 6 2)
        \\(+ (* 2 4) (- 4 6))
        \\(define a 3)
        \\(define b (+ a 1))
        \\(+ a b (* a b))
        \\(= a b)
        \\(if (and (> b a) (< b (* a b)))
        \\b
        \\a)
        \\(cond ((= a 4) 6)
        \\((= b 4) (+ 6 7 a))
        \\(else 25))
        \\(+ 2 (if (> b a) b a))
        \\(* (cond ((> a b) a)
        \\((< a b) b)
        \\(else -1))
        \\(+ a 1))
        \\
    ;
}
