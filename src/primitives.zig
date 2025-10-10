const std = @import("std");
const Ast = @import("Ast.zig");
const Parser = @import("Parser.zig");
const Env = Ast.Env;
const Expr = Ast.Expr;

pub const Primitive = std.meta.DeclEnum(methods);
pub const tags = std.meta.tags(Primitive);
pub const tags_len = @typeInfo(Primitive).@"enum".fields.len;
pub const Func = @TypeOf(&methods.eval);

pub fn getFunc(prim: Primitive) Func {
    return switch (prim) {
        inline else => |t| return @field(methods, @tagName(t)),
    };
}

fn argsAs(T: type, comptime field: Ast.Expr.Tag, args: Ast.Iterator) struct { T, Ast.Iterator } {
    if (false) {
        var iter = args;
        while (iter.next()) |arg| {
            std.debug.print("{f}\n", .{arg.iterator(args.ast)});
        }
    }
    switch (@typeInfo(T)) {
        .array => |a| {
            var ret: T = undefined;
            var iter = args;
            for (0..a.len) |i| ret[i] = iter.next().?.as(field);
            return .{ ret, iter };
        },
        else => unreachable,
    }
}

const methods = struct {
    const Ret = Ast.Error!RetRest;
    const RetRest = struct { id: Expr.Id, rest: Expr.Id };

    fn err(args: Ast.Iterator, env: Expr.Id, e: Ast.Error) Ret {
        _ = args; // autofix
        _ = env; // autofix
        std.debug.print("TODO log error location\n", .{});
        return e;
    }

    pub fn @"+"(args: Ast.Iterator, env: Expr.Id) Ret {
        _ = env; // autofix
        const res, const rest = argsAs([2]f64, .num, args);
        const b = try args.ast.createExpr(.{ .num = res[0] + res[1] });
        return .{ .id = b.id, .rest = rest.id };
    }

    pub fn eval(args: Ast.Iterator, env: Expr.Id) Ret {
        var iter = args;
        const root = iter.next() orelse return err(args, env, error.Eval);
        // std.debug.print("eval() root: {}\n{f}", .{ root.expr().*, ast });

        switch (root.expr().*) {
            .cons => |c| {
                if (c.isEmpty()) return .{ .id = root.id, .rest = .nil };
                var citer = root;
                const fst = citer.next() orelse return err(args, env, error.Eval);
                const fst_e = fst.expr().*;
                // std.debug.print("eval() cons fst: {} {t}\n{f}", .{ fstid, fst, ast });
                if (fst_e == .sym) {
                    if (std.meta.stringToEnum(Primitive, fst_e.sym.str(root.ast))) |prim| {
                        return getFunc(prim)(citer, env);
                    }
                }
                return err(args, env, error.Eval);
            },
            else => |x| std.debug.panic("TODO {t}\n", .{x}),
        }
        unreachable;
    }

    pub fn cons(args: Ast.Iterator, env: Expr.Id) Ret {
        _ = env; // autofix
        var iter = args;
        const e = try args.ast.createExpr(.{ .cons = .{ .fst = iter.next().?.id, .snd = .nil } });
        return .{ .id = e.id, .rest = iter.id };
    }

    pub fn car(args: Ast.Iterator, env: Expr.Id) Ret {
        _ = env; // autofix
        var iter = args;
        return .{ .id = iter.next().?.id, .rest = iter.id };
    }

    test @"+" {
        var ast = try Parser.parse("(+ 1 2)", t_gpa);
        defer ast.deinit();
        const x = try methods.eval(ast.root_list.iterator(&ast), ast.root_env);
        try testing.expectEqual(3, x.id.iterator(&ast).as(.num));
    }

    test car {
        var ast = try Parser.parse("(car (1 2))", t_gpa);
        defer ast.deinit();
        const x = try methods.eval(ast.root_list.iterator(&ast), ast.root_env);
        var iter = x.id.iterator(&ast);
        try testing.expectEqual(1, iter.next().?.as(.num));
    }
};

// const _primitives = arr: {
//     var arr = std.EnumArray(Primitive, Expr.Fun).init(.{});
//     for (std.meta.declarations(primitives), &arr) |p, *e| {
//         e.* = std.meta.stringToEnum(Primitive, p.name).?;
//     }
//     break :arr arr;
// };

// const PrimitivesArray = [prim_tags_len]*const Expr.Fun;

const testing = std.testing;
const t_gpa = testing.allocator;

test methods {
    _ = methods;
}
