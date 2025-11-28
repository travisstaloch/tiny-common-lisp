//!
//! adapted / inspired by
//!   https://github.com/Robert-van-Engelen/tinylisp and
//!   https://github.com/daneelsan/tinylisp/
//!

const Interp = @This();

hp: u32,
sp: u32,
memory: []Expr,
tru: Expr,
env: Expr,
w: *Io.Writer,
print_mode: PrintMode,

const PrintMode = enum {
    quiet,
    /// print loop intermediate results in run()
    repl,
};

pub fn init(cell: []Expr, w: *Io.Writer, print_mode: PrintMode) Interp {
    var p: Interp = .{
        .memory = cell,
        .hp = 0,
        .sp = @intCast(cell.len),
        .tru = undefined,
        .env = undefined,
        .w = w,
        .print_mode = print_mode,
    };
    @memset(p.memory, .empty);

    p.tru = p.atom("t");
    p.env = p.pair(p.tru, p.tru, nil);
    for (0..prims_len) |i| {
        const pr: Prim = @enumFromInt(i);
        p.env = p.pair(p.atom(@tagName(pr)), box(.prim, @intCast(i)), p.env);
    }
    return p;
}

pub const Expr = extern union {
    float: f64,
    int: u64,
    boxed: packed struct(u64) {
        payload: u48,
        tag: Tag,
        /// tag == .cons and sign == 1 indicates top level list.  all other
        /// boxed values have sign == 0.
        sign: u1 = 0,
    },

    pub const empty: Expr = .{ .int = 0 };

    pub const Tag = enum(u15) {
        atom = 0x7ff8, // 0b0111111111111_000
        prim = 0x7ff9, // 0b0111111111111_001
        cons = 0x7ffa, // 0b0111111111111_010
        clos = 0x7ffb, // 0b0111111111111_011
        macr = 0x7ffc, // 0b0111111111111_100
        nil = 0x7ffd, //  0b0111111111111_101

        pub fn int(t: Tag) u16 {
            return @intFromEnum(t);
        }

        pub fn isOneOf(t: Tag, comptime tags: []const u16) bool {
            const V = @Vector(tags.len, u16);
            const wanted: V = @bitCast(tags[0..tags.len].*);
            const actual: [tags.len]u16 = @splat(@intFromEnum(t));
            return @reduce(.Or, wanted == actual);
        }

        pub fn pair(a: Tag, b: Tag) u32 {
            return @bitCast([2]u16{ a.int(), b.int() });
        }
    };

    pub fn ord(x: Expr) u48 {
        return x.boxed.payload;
    }

    pub fn not(x: Expr) bool {
        return x.boxed.tag.int() == NIL_;
    }

    pub fn equ(x: Expr, y: Expr) bool {
        return x.int == y.int;
    }

    pub fn equStructural(a: Expr, b: Expr, p: *Interp) bool {
        return switch (a.boxed.tag.pair(b.boxed.tag)) {
            Expr.Tag.pair(.atom, .atom) => std.mem.eql(u8, p.atomName(a), p.atomName(b)),
            Expr.Tag.pair(.nil, .nil) => true,
            Expr.Tag.pair(.cons, .cons) => p.carAssume(a).equStructural(p.carAssume(b), p) and
                p.cdrAssume(a).equStructural(p.cdrAssume(b), p),
            else => a.int == b.int,
        };
    }

    pub fn tagName(x: Expr) []const u8 {
        return if (x.isTagged()) @tagName(x.boxed.tag) else "num";
    }

    pub fn isTagged(x: Expr) bool {
        return switch (x.boxed.tag.int()) {
            ATOM, PRIM, CONS, CLOS, MACR, NIL_ => true,
            else => false,
        };
    }

    pub const Fmt = struct {
        p: *Interp,
        e: Expr,

        pub fn format(f: Fmt, w: *Io.Writer) !void {
            return f.formatFallible(w) catch return error.WriteFailed;
        }
        fn formatList(f: Fmt, w: *Io.Writer, t: *Expr) !void {
            while (true) {
                try w.print("{f}", .{(try f.p.car(t.*)).fmt(f.p)});
                t.* = try f.p.cdr(t.*);
                switch (t.boxed.tag.int()) {
                    NIL_ => break,
                    CONS => {},
                    else => {
                        try w.print(" . {f}", .{t.fmt(f.p)});
                        break;
                    },
                }
                try w.writeAll(" ");
            }
        }
        fn formatFallible(f: Fmt, w: *Io.Writer) !void {
            switch (f.e.boxed.tag.int()) {
                NIL_ => try w.writeAll("()"),
                ATOM => try w.writeAll(f.p.atomName(f.e)),
                PRIM => try w.print("<{t}>", .{@as(Prim, @enumFromInt(ord(f.e)))}),
                CONS => {
                    var t = f.e;
                    const is_root = 1 - f.e.boxed.sign; // sign set to 1 for top level list
                    if (f.p.isQuote(t)) {
                        try w.writeByte('\'');
                        t = f.p.cdrAssume(t);
                        try formatList(f, w, &t);
                    } else {
                        try w.writeAll("("[0..is_root]);
                        try formatList(f, w, &t);
                        try w.writeAll(")"[0..is_root]);
                    }
                },
                CLOS => try w.print("{{{d}}}", .{f.e.ord()}),
                MACR => try w.print("[{d}]", .{f.e.ord()}),
                else => try w.print("{d}", .{f.e.float}),
            }
        }
    };

    pub fn fmt(e: Expr, p: *Interp) Fmt {
        return .{ .e = e, .p = p };
    }

    // TODO check tag
    /// convert or check number n
    pub fn toNum(e: Expr) Expr {
        return e;
    }
};

const ATOM = Expr.Tag.atom.int();
const PRIM = Expr.Tag.prim.int();
const CONS = Expr.Tag.cons.int();
const CLOS = Expr.Tag.clos.int();
const MACR = Expr.Tag.macr.int();
const NIL_ = Expr.Tag.nil.int();

pub const nil = box(.nil, 0);

pub fn heap(i: *Interp) [*:0]u8 {
    return @ptrCast(i.memory);
}

pub fn box(t: Expr.Tag, payload: u48) Expr {
    return .{ .boxed = .{ .tag = t, .payload = payload } };
}

pub fn abort(_: *Interp, comptime fmt: []const u8, args: anytype) noreturn {
    std.debug.print(fmt, args);
    std.process.exit(1);
}

pub fn checkStack(p: *Interp) void {
    if (p.hp > (p.sp * @sizeOf(Expr))) {
        @panic("stack overflow");
    }
}

fn strcmp(a: [*:0]const u8, b: []const u8) bool {
    return std.mem.eql(u8, a[0..b.len], b) and a[b.len] == 0;
}

pub fn atom(p: *Interp, s: []const u8) Expr {
    var i: u32 = 0;
    while (i < p.hp) {
        if (strcmp(p.heap()[i..], s)) {
            return box(.atom, i);
        }
        const len = std.mem.len(p.heap() + i);
        assert(p.heap()[i + len] == 0);
        i += @intCast(len + 1);
    }
    if (i == p.hp) {
        @memcpy(p.heap() + i, s);
        p.heap()[i + s.len] = 0;
        p.hp += @intCast(s.len + 1);
        p.checkStack();
    }
    return box(.atom, i);
}

pub fn atomName(p: *Interp, x: Expr) []const u8 {
    assert(x.boxed.tag == .atom);
    return std.mem.sliceTo(p.heap()[x.ord()..], 0);
}

pub fn cons(p: *Interp, x: Expr, y: Expr) Expr {
    p.sp -= 1;
    p.memory[p.sp] = x;
    p.sp -= 1;
    p.memory[p.sp] = y;
    p.checkStack();
    return box(.cons, p.sp);
}

pub fn pair(p: *Interp, v: Expr, x: Expr, e: Expr) Expr {
    return p.cons(p.cons(v, x), e);
}

pub fn isQuote(p: *Interp, t: Expr) bool {
    return p.carAssume(t).boxed.tag == .atom and
        std.mem.eql(u8, "quote", p.atomName(p.carAssume(t)));
}

/// parse src and then loop evaluating intermediates
pub fn run(p: *Interp, src: [:0]const u8, file_path: []const u8) Error!Expr {
    var t = try p.parse(src, file_path);
    var result = nil;
    while (t.boxed.tag == .cons) {
        result = try p.eval(try p.car(t), p.env);
        if (p.print_mode == .repl)
            try p.w.print("{f}\n", .{result.fmt(p)});
        t = try p.cdr(t);
    }
    if (p.print_mode == .repl) try p.w.flush();
    return result;
}

pub const Error = error{ Parse, NonPair, Unbound, CannotApply, User } ||
    Io.Reader.Error ||
    Io.Writer.Error ||
    std.fmt.ParseFloatError ||
    std.fs.File.OpenError ||
    std.fs.File.ReadError ||
    std.mem.Allocator.Error;

pub fn parse(l: *Interp, src: [:0]const u8, file_path: []const u8) Error!Expr {
    // trace("parse() called", .{});
    var t = Tokenizer{
        .src = src,
        .pos = 0,
        .file_path = file_path,
    };

    var ret = try l.parseList(&t, .eof);
    assert(ret.boxed.tag == .cons);
    ret.boxed.sign = 1; // sign=1 means top level list
    return ret;
}

fn parseExpr(p: *Interp, t: *Tokenizer, token: Tokenizer.Token) Error!Expr {
    return p.parseExprInner(t, token) catch |e|
        p.err(e, "{f}", .{token.fmt(t)});
}
fn parseExprInner(p: *Interp, t: *Tokenizer, token: Tokenizer.Token) Error!Expr {
    // trace("parseExpr({f})", .{token.fmt(t)});
    const src = token.src(t.src);

    return switch (token.tag) {
        .lparen => try p.parseList(t, .rparen),
        .symbol, .string => if (std.mem.eql(u8, src, "nil"))
            nil
        else
            p.atom(src),
        .number => .{ .float = try std.fmt.parseFloat(f64, src) },
        .eof => nil,
        .quote => {
            const quoted = try p.parseExprInner(t, t.next());
            return p.cons(p.atom("quote"), p.cons(quoted, nil));
        },
        .rparen => return error.Parse,
        else => |tag| std.debug.panic("TODO handle {t}", .{tag}),
    };
}

fn parseList(p: *Interp, t: *Tokenizer, comptime end_tag: Tokenizer.Token.Tag) Error!Expr {
    const token = t.next();
    // trace("parseList() called {f}", .{token.fmt(t)});
    switch (token.tag) {
        end_tag => return nil,
        .dot => {
            const last_expr = try p.parseExprInner(t, t.next());
            const right_paren = t.next();
            if (right_paren.tag != .rparen) return error.Parse;
            return last_expr;
        },
        else => {
            // std.debug.panic("TODO handle {t}", .{tag}),
            const first = try p.parseExprInner(t, token);
            const rest = try p.parseList(t, end_tag);
            return p.cons(first, rest);
        },
    }
}

pub fn err(p: *Interp, e: Error, comptime fmt: []const u8, args: anytype) Error {
    // TODO file:line:col
    try p.w.print("error.{t}: " ++ fmt ++ "\n", .{e} ++ args);
    return e;
}

pub fn gcOld(p: *Interp) void {
    p.sp = @intCast(p.env.ord());
}

fn carOpt(p: *Interp, x: Expr) ?Expr {
    return if (x.boxed.tag.isOneOf(&.{ CONS, CLOS, MACR }))
        p.memory[x.ord() + 1]
    else
        null;
}

fn car(p: *Interp, x: Expr) Error!Expr {
    return p.carOpt(x) orelse p.err(error.NonPair, "{f}", .{x.fmt(p)});
}

// TODO replace car with carAssume whenever CONS/CLOS/MACR is known
fn carAssume(p: *Interp, x: Expr) Expr {
    return p.carOpt(x).?;
}

fn cdrOpt(p: *Interp, x: Expr) ?Expr {
    return if (x.boxed.tag.isOneOf(&.{ CONS, CLOS, MACR }))
        p.memory[x.ord()]
    else
        null;
}

fn cdr(p: *Interp, x: Expr) Error!Expr {
    return p.cdrOpt(x) orelse p.err(error.NonPair, "{f}", .{x.fmt(p)});
}

fn cdrAssume(p: *Interp, x: Expr) Expr {
    return p.cdrOpt(x).?;
}

/// construct a closure, returns a boxed CLOS
fn closure(p: *Interp, vars: Expr, body: Expr, e: Expr) Error!Expr {
    const pr = p.pair(vars, body, if (e.equ(p.env)) nil else e);
    return box(.clos, pr.ord());
}

fn macro(p: *Interp, v: Expr, x: Expr) Error!Expr {
    return box(.macr, p.cons(v, x).ord());
}

/// look up a symbol in an environment, return its value or ERR if not found
fn assoc(p: *Interp, a: Expr, e0: Expr) Error!Expr {
    var e = e0;
    while (e.boxed.tag.int() == CONS and !a.equ(try p.car(try p.car(e)))) {
        e = try p.cdr(e);
    }
    return if (e.boxed.tag.int() == CONS)
        try p.cdr(try p.car(e))
    else
        p.err(error.Unbound, "{f}", .{a.fmt(p)});
}

fn let(p: *Interp, x: Expr) bool {
    return !x.not() and !(if (p.cdrOpt(x)) |y| y.not() else true);
}

// create environment by extending `env` with variables `vars` bound to values `vals`
// &rest is a parameter pattern that allows a function to accept a variable
// number of arguments, collecting all remaining arguments into a list.
fn bind(p: *Interp, vars: Expr, vals: Expr, e: Expr) Error!Expr {
    trace("bind vars {f}", .{vars.fmt(p)});
    return switch (vars.boxed.tag.int()) {
        NIL_ => e,
        CONS => blk: {
            const v = try p.car(vars);
            trace("bind v {f}", .{v.fmt(p)});
            if (v.boxed.tag == .atom and std.mem.eql(u8, "&rest", p.atomName(v))) {
                // collect remaining args into a list
                const rest = try p.car(try p.cdr(vars));
                return p.pair(rest, vals, e);
            }

            break :blk p.bind(
                try p.cdr(vars),
                try p.cdr(vals),
                p.pair(v, try p.car(vals), e),
            );
        },
        else => p.pair(vars, vals, e),
    };
}

/// return a new list of evaluated Lisp expressions t in environment e
fn evlis(p: *Interp, t: Expr, e: Expr) Error!Expr {
    return switch (t.boxed.tag.int()) {
        CONS => p.cons(
            try p.eval(p.carAssume(t), e),
            try p.evlis(p.cdrAssume(t), e),
        ),
        ATOM => p.assoc(t, e),
        else => nil,
    };
}

/// apply closure `clos` to arguments `args` in environment `e`
pub fn reduce(p: *Interp, clos: Expr, args: Expr, e: Expr) Error!Expr {
    const clos_fun = try p.car(clos);
    const clos_env = try p.cdr(clos);
    const clos_vars = try p.car(clos_fun);
    const clos_body = try p.cdr(clos_fun);
    const eval_args = try p.evlis(args, e);
    return p.eval(clos_body, try p.bind(
        clos_vars,
        eval_args,
        if (clos_env.not()) p.env else clos_env,
    ));
}

fn callPrim(pm: Prim, p: *Interp, t: Expr, e: Expr) Error!Expr {
    // trace("callPrim({t}, {f})", .{ pm, t.fmt(p) });
    return switch (pm) {
        inline else => |tag| try @field(primitives, @tagName(tag))(p, t, e),
    };
}

fn doApply(p: *Interp, f: Expr, t: Expr, e: Expr) Error!Expr {
    trace("doApply {f} {f}", .{ f.fmt(p), t.fmt(p) });
    return switch (f.boxed.tag.int()) {
        PRIM => try callPrim(@enumFromInt(f.ord()), p, t, e),
        CLOS => try p.reduce(f, t, e),
        MACR => blk: {
            var d = e;
            var x = t;
            var v = try p.car(f);
            while (v.boxed.tag == .cons) {
                d = p.pair(try p.car(v), try p.car(x), d);
                trace("doApply macr x {f} v {f}", .{ x.fmt(p), v.fmt(p) });
                x = try p.cdr(x);
                v = try p.cdr(v);
            }
            if (v.boxed.tag == .atom) d = p.pair(v, x, d);
            const expanded = try p.eval(try p.cdr(f), d);
            break :blk try p.eval(expanded, e);
        },
        else => p.err(error.CannotApply, "{s} {f}", .{ f.tagName(), f.fmt(p) }),
    };
}

fn eval(p: *Interp, t: Expr, e: Expr) Error!Expr {
    errdefer trace("eval error on {f}", .{t.fmt(p)});
    const ret = switch (t.boxed.tag.int()) {
        ATOM => try p.assoc(t, e),
        CONS => try p.doApply(
            try p.eval(try p.car(t), e),
            try p.cdr(t),
            e,
        ),
        MACR => {
            // bind macro f variables v to the given arguments literally (i.e.
            // without evaluating the arguments)
            var d = e;
            const f = try p.eval(try p.car(t), e);
            var v = try p.car(f);
            var x = try p.cdr(t);
            while (v.boxed.tag.int() == CONS) {
                d = p.pair(try p.car(v), try p.car(x), d);
                v = try p.cdr(v);
                x = try p.cdr(x);
            }
            if (v.boxed.tag.int() == ATOM) {
                d = p.pair(v, x, d);
            }
            // expand macro f, then continue evaluating the expanded x
            return try p.eval(try p.cdr(f), d);
        },
        else => t,
    };
    trace("eval\n  {f}\n  => {f}", .{ t.fmt(p), ret.fmt(p) });
    return ret;
}

pub fn fromBool(p: *Interp, b: bool) Expr {
    return if (b) p.tru else nil;
}

pub const Prim = std.meta.DeclEnum(primitives);
pub const prims_len = @typeInfo(Prim).@"enum".fields.len;

// TODO maybe accept `p: *primitives` instead of *Interp and do
// @fieldParentPtr mixins
const primitives = struct {
    /// (eval x) return evaluated x (such as when x was quoted)
    pub fn eval(p: *Interp, t: Expr, e: Expr) Error!Expr {
        return try p.eval(try p.car(try p.evlis(t, e)), e);
    }
    /// (quote x) special form, returns x unevaluated "as is"
    pub fn quote(p: *Interp, t: Expr, _: Expr) Error!Expr {
        if (!(try p.cdr(t)).not()) return p.err(error.User, "quote expected one arg. found {f}", .{t.fmt(p)});
        return try p.car(t);
    }
    /// (cons x y) construct pair (x . y)
    pub fn cons(p: *Interp, t: Expr, e: Expr) Error!Expr {
        const t1 = try p.evlis(t, e);
        return p.cons(try p.car(t1), try p.car(try p.cdr(t1)));
    }
    /// (car p) car of pair p
    pub fn car(p: *Interp, t: Expr, e: Expr) Error!Expr {
        return try p.car(try p.car(try p.evlis(t, e)));
    }
    /// (cdr p) cdr of pair p
    pub fn cdr(p: *Interp, t: Expr, e: Expr) Error!Expr {
        return try p.cdr(try p.car(try p.evlis(t, e)));
    }
    // (list x1 x2 ... xk)
    // returns the list of `x1`, `x2`, ..., `xk`.  That is, `(x1 x2 ... xk)` with all `x` evaluated.
    pub fn list(p: *Interp, t: Expr, e: Expr) Error!Expr {
        return p.evlis(t, e);
    }
    /// (apply f arg1 arg2 ... argn-list)
    /// argn-list is a list
    pub fn apply(p: *Interp, t: Expr, e: Expr) Error!Expr {
        if (t.not()) return p.err(error.User, "apply missing function arg", .{});

        var f = try p.car(t);
        if (f.boxed.tag == .cons and p.isQuote(f))
            f = p.carAssume(p.cdrAssume(f));

        f = try p.eval(f, e);
        if (f.boxed.tag == .atom)
            f = try p.assoc(f, p.env);

        const args = try p.evlis(try p.cdr(t), e); // args = (arg1 arg2 ... argn list)
        if (args.not()) return try p.doApply(f, args, e);

        // build argument list
        var args2 = nil;
        var tail: ?Expr = null; // last cons cell for appending
        var current = args;
        while (current.boxed.tag == .cons) {
            const next = try p.cdr(current);
            if (next.boxed.tag != .cons) {
                const list_part = try p.car(current);
                if (tail) |tl|
                    // append list_part to the end of args2
                    p.memory[tl.ord()] = list_part
                else
                    // no individual args, args2 is just the list
                    args2 = list_part;

                break;
            }
            // add the current arg to the end of args2
            const new_cons = p.cons(try p.car(current), nil);
            if (tail) |tl|
                p.memory[tl.ord()] = new_cons
            else
                args2 = new_cons;

            tail = new_cons;
            current = next;
        }

        return try p.doApply(f, args2, e);
    }
    /// (funcall function args)
    /// applies function to args. If function is a symbol, it is coerced to a
    /// function as if by finding its functional value
    pub fn funcall(p: *Interp, t: Expr, e: Expr) Error!Expr {
        if (t.not()) return p.err(error.User, "funcall missing function arg", .{});
        var f = try p.car(t);
        if (f.boxed.tag == .cons and p.isQuote(f))
            f = p.carAssume(p.cdrAssume(f));
        f = try p.eval(f, e);
        if (f.boxed.tag == .atom)
            f = try p.assoc(f, p.env);
        // Evaluate all arguments normally (no list splicing like apply)
        const args = try p.evlis(try p.cdr(t), e);
        return try p.doApply(f, args, e);
    }
    const Op = enum { add, sub, mul, div };
    fn mathOp(p: *Interp, t: Expr, e: Expr, comptime op: Op) Error!Expr {
        var t1 = try p.evlis(t, e);
        var n = try p.car(t1);
        while (true) {
            t1 = try p.cdr(t1);
            if (t1.not()) break;
            switch (op) {
                .add => n.float += (try p.car(t1)).toNum().float,
                .sub => n.float -= (try p.car(t1)).toNum().float,
                .mul => n.float *= (try p.car(t1)).toNum().float,
                .div => n.float /= (try p.car(t1)).toNum().float,
            }
        }
        return n.toNum();
    }
    /// (+ n1 n2 ... nk) sum of n1 to nk
    pub fn @"+"(p: *Interp, t: Expr, e: Expr) Error!Expr {
        return mathOp(p, t, e, .add);
    }
    /// (- n1 n2 ... nk) sum of n1 to nk
    pub fn @"-"(p: *Interp, t: Expr, e: Expr) Error!Expr {
        return mathOp(p, t, e, .sub);
    }
    /// (* n1 n2 ... nk) sum of n1 to nk
    pub fn @"*"(p: *Interp, t: Expr, e: Expr) Error!Expr {
        return mathOp(p, t, e, .mul);
    }
    /// (/ n1 n2 ... nk) sum of n1 to nk
    pub fn @"/"(p: *Interp, t: Expr, e: Expr) Error!Expr {
        return mathOp(p, t, e, .div);
    }
    /// (truncate n) integer part of n, round toward 0
    pub fn truncate(p: *Interp, t: Expr, e: Expr) Error!Expr {
        const n = try p.car(try p.evlis(t, e));
        // @floatFromInt(@as(u128, @intFromFloat(n.float)))
        return .{ .float = @trunc(n.float) };
    }
    fn cmpOp(p: *Interp, t: Expr, e: Expr, comptime cmp: enum { lt, gt, eq, le, ge }) Error!Expr {
        const t1 = try p.evlis(t, e);
        const x = try p.car(t1);
        const y = try p.car(try p.cdr(t1));
        return p.fromBool(switch (cmp) {
            .lt => x.float - y.float < 0,
            .gt => x.float - y.float > 0,
            .eq => x.equ(y),
            .le => x.float - y.float <= 0,
            .ge => x.float - y.float >= 0,
        });
    }
    /// (< n1 n2) t if n1<n2, otherwise ()
    pub fn @"<"(p: *Interp, t: Expr, e: Expr) Error!Expr {
        return cmpOp(p, t, e, .lt);
    }
    /// (> n1 n2) t if n1>n2, otherwise ()
    pub fn @">"(p: *Interp, t: Expr, e: Expr) Error!Expr {
        return cmpOp(p, t, e, .gt);
    }
    /// (= n1 n2) t if n1=n2, otherwise ()
    pub fn @"="(p: *Interp, t: Expr, e: Expr) Error!Expr {
        return cmpOp(p, t, e, .eq);
    }
    /// (<= n1 n2) t if n1<=n2, otherwise ()
    pub fn @"<="(p: *Interp, t: Expr, e: Expr) Error!Expr {
        return cmpOp(p, t, e, .le);
    }
    /// (>= n1 n2) t if n1>=n2, otherwise ()
    pub fn @">="(p: *Interp, t: Expr, e: Expr) Error!Expr {
        return cmpOp(p, t, e, .ge);
    }
    /// (or x1 x2 ... xk) first x that is truthy, otherwise ()
    pub fn @"or"(p: *Interp, t: Expr, e: Expr) Error!Expr {
        var x = Interp.nil;
        var t1 = t;
        while (true) {
            if (t1.not()) break;
            x = try p.eval(try p.car(t1), e);
            if (!x.not()) break;
            t1 = try p.cdr(t1);
        }
        return x;
    }
    /// (and x1 x2 ... xk) last x if all x are truthy, otherwise ()
    pub fn @"and"(p: *Interp, t: Expr, e: Expr) Error!Expr {
        var x = Interp.nil;
        var t1 = t;
        while (true) {
            if (t1.not()) break;
            x = try p.eval(try p.car(t1), e);
            if (x.not()) break;
            t1 = try p.cdr(t1);
        }
        return x;
    }
    /// (if x y z) if x then y else z
    pub fn @"if"(p: *Interp, t: Expr, e: Expr) Error!Expr {
        const cnd = try p.eval(try p.car(t), e);
        const branches = try p.cdr(t);
        const res = try p.car(if (cnd.not()) try p.cdr(branches) else branches);
        // trace("if cnd {f} branches {f} res {f}", .{ cnd.fmt(p), branches.fmt(p), res.fmt(p) });
        return try p.eval(res, e);
    }
    /// (defvar v x) define a named value globally
    pub fn defvar(p: *Interp, t: Expr, e: Expr) Error!Expr {
        const t1 = try p.cdr(t);
        const x = try p.car(t1);
        const t2 = try p.eval(x, e);
        // trace("fst {f} t1 {f} x {f} t2 {f}", .{ (try p.car(t)).fmt(p), t1.fmt(p), x.fmt(p), t2.fmt(p) });
        p.env = p.pair(try p.car(t), t2, p.env);
        return try p.car(t);
    }
    // (defun name params body) define a named function globally
    pub fn defun(p: *Interp, t: Expr, e: Expr) Error!Expr {
        const name = try p.car(t);
        const params = try p.car(try p.cdr(t));
        const body = try p.car(try p.cdr(try p.cdr(t)));
        trace("defun name:{f} params:{f} body:{f}", .{ name.fmt(p), params.fmt(p), body.fmt(p) });
        p.env = p.pair(name, try p.closure(params, body, e), p.env);
        return name;
    }
    pub fn cond(p: *Interp, t0: Expr, e: Expr) Error!Expr {
        var t = t0;
        while ((try p.eval(try p.car(try p.car(t)), e)).not()) {
            t = try p.cdr(t);
        }
        return try p.car(try p.cdr(try p.car(t)));
    }
    /// (not x) t if x is (), otherwise ()
    pub fn not(p: *Interp, t: Expr, e: Expr) Error!Expr {
        return p.fromBool((try p.car(try p.evlis(t, e))).not());
    }
    /// (lambda v x) construct a closure
    pub fn lambda(p: *Interp, t: Expr, e: Expr) Error!Expr {
        return p.closure(try p.car(t), try p.car(try p.cdr(t)), e);
    }
    /// (let* ((k1 v1) (k2 v2) ... (kk vk)) y1 ... yn)
    /// eval `y`s with `kv`s in e0
    pub fn @"let*"(p: *Interp, t: Expr, e0: Expr) Error!Expr {
        var kvs = try p.car(t);
        var kv = try p.car(kvs);
        var e = e0;
        while (p.let(kv)) {
            e = p.pair(
                try p.car(kv),
                try p.eval(try p.car(try p.cdr(kv)), e),
                e,
            );
            kvs = p.cdrOpt(kvs) orelse break;
            kv = p.carOpt(kvs) orelse break;
        }

        var current = p.cdrOpt(t) orelse return Interp.nil;
        var result = Interp.nil;
        while (current.boxed.tag.int() == CONS) {
            result = try p.eval(p.carAssume(current), e);
            current = p.cdrAssume(current);
        }
        return result;
    }
    pub fn consp(p: *Interp, t: Expr, e: Expr) Error!Expr {
        const x = try p.car(try p.evlis(t, e));
        return p.fromBool(x.boxed.tag == .cons);
    }
    /// (progn (x1 x2 ...)) evaluates each x$i, returning the last one
    pub fn progn(p: *Interp, t: Expr, e: Expr) Error!Expr {
        var result = Interp.nil;
        var current = t;
        while (current.boxed.tag == .cons) {
            result = try p.eval(try p.car(current), e);
            current = try p.cdr(current);
        }
        return result;
    }
    pub fn print(p: *Interp, t: Expr, e: Expr) Error!Expr {
        const t1 = try p.car(try p.evlis(t, e));
        try p.w.print("{f}\n", .{t1.fmt(p)});
        try p.w.flush();
        return t1;
    }
    pub fn load(p: *Interp, t: Expr, _: Expr) Error!Expr {
        const t1 = try p.car(t);
        if (t1.boxed.tag != .atom)
            return p.err(error.User, "expected atom. found {f}", .{t1.fmt(p)});
        const name = p.atomName(t1);
        trace("load name {s}\n", .{name});
        const f = try std.fs.cwd().openFile(name, .{});
        defer f.close();
        const len = try f.getEndPos();
        const mem = p.heap()[p.hp..][0 .. len + 1];
        const amt = try f.read(mem);
        assert(amt == len);
        mem[len] = 0;
        return try p.run(mem[0..len :0], name);
    }
    pub fn macro(p: *Interp, t: Expr, _: Expr) Error!Expr {
        return p.macro(try p.car(t), try p.car(try p.cdr(t)));
    }
    /// object equality - whether two objects are the same object in memory
    pub fn eq(p: *Interp, t: Expr, e: Expr) Error!Expr {
        const a = try p.eval(try p.car(t), e);
        const b = try p.eval(try p.car(try p.cdr(t)), e);
        return p.fromBool(if (a.isTagged() and b.isTagged())
            a.ord() == b.ord()
        else
            a.int == b.int);
    }
    /// structural equality
    pub fn equal(p: *Interp, t: Expr, e: Expr) Error!Expr {
        const a = try p.eval(try p.car(t), e);
        const b = try p.eval(try p.car(try p.cdr(t)), e);
        return p.fromBool(a.equStructural(b, p));
    }

    test @"let*" {
        try testEval("3",
            \\(let* ((x 1) (y 2)) (let* ((z (+ x y))) z))
        );
        try testEval("3",
            \\(let* ((x 2)) ((lambda (y) (let* ((f '+) (x 1)) (apply f (list x y)))) x))
        );
    }
    test load {
        try testEval("6", "(load examples/basic.lisp)");
    }
    test list {
        try testEval("(1 2)", "(list 1 2)");
    }
    test "macro" {
        try testEval("4",
            \\(defvar defun-macro (macro (f v x) (list 'defvar f (list 'lambda v x))))
            \\(defun-macro double (x) (+ x x))
            \\(double 2)
        );
    }
    test "= eq equal" {
        try testEval("t", "(eq 3 3)");
        try testEval("t", "(= 3 3)");
        try testEval("()", "(eq '(1 2) '(1 2))"); // different list objects
        try testEval("t", "(equal '(5) '(5))");
        try testEval("t", "(equal '(1 2) '(1 2))");
    }
    test apply {
        try testEval("6", "(apply '+ '(1 2 3))");
        try testEval("10", "(apply '+ 1 2 '(3 4))");
    }
    test defun {
        try testEval("4", "(defun double (x) (+ x x)) (double 2)");
    }
    test "&rest" {
        try testEval("(1 2 (3 4 5))",
            \\(defun rest-as-list (a b &rest rest)
            \\  (list a b rest))
            \\(rest-as-list 1 2 3 4 5)
        );
    }
};

fn trace(comptime fmt: []const u8, args: anytype) void {
    const log = std.log.scoped(.tinylisp);
    if (false)
        log.debug(fmt, args);

    if (@import("build-options").trace)
        std.debug.print(fmt ++ "\n", args);
}

const std = @import("std");
const Io = std.Io;
const File = std.fs.File;
const assert = std.debug.assert;
const Tokenizer = @import("Tokenizer.zig");

const testing = std.testing;
const t_gpa = testing.allocator;

fn expectExprEqual(expected: Expr.Fmt, actual: Expr.Fmt) !void {
    // trace("{f}\n{f}\n", .{ expected, actual });
    try testing.expectEqual(expected.e.boxed.tag, actual.e.boxed.tag);
    const ep = expected.p;
    const ap = actual.p;
    switch (expected.e.boxed.tag.int()) {
        ATOM => try testing.expectEqualStrings(
            ep.atomName(expected.e),
            ap.atomName(actual.e),
        ),
        CONS => {
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
    try testParseFmt(.{ .file_path = "tests/dotcall.lisp" });
}

fn testEval(expected: []const u8, src: [:0]const u8) !void {
    var discarding = std.Io.Writer.Discarding.init(&.{});
    const N = 4096 * 4;
    var memory: [N]Expr = undefined;
    var l: Interp = .init(&memory, &discarding.writer, .quiet);
    const e = try l.run(src, "<testEval>");
    try testing.expectFmt(expected, "{f}", .{e.fmt(&l)});
}

test {
    _ = primitives;
}
test "nil sym" {
    try testEval("()", "nil");
}
