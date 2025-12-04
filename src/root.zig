//!
//! adapted from
//!   https://github.com/Robert-van-Engelen/tinylisp
//!   https://github.com/daneelsan/tinylisp/
//! references
//!   https://github.com/cryptocode/bio
//!   https://github.com/tekknolagi/blog-lisp/tree/a02545444f34c1eb785490827d4197141baac113
//!   https://github.com/jvk36/RMM
//!   https://github.com/eudoxia0/interim
//!

const Interp = @This();

nil: Expr.Id,
tru: Expr.Id,
env: Expr.Id,
w: *Io.Writer,
print_mode: PrintMode,
regions: std.ArrayList(Region),
root_checkpoint: Region.Header,
/// string -> index into exprs
atoms: std.StringArrayHashMapUnmanaged(Expr.Id),
exprs: std.ArrayList(Expr),

pub const Error = error{ Parse, NonPair, Unbound, CannotApply, User } ||
    Io.Reader.Error ||
    Io.Writer.Error ||
    std.fmt.ParseFloatError ||
    std.fs.File.OpenError ||
    std.fs.File.ReadError ||
    std.mem.Allocator.Error;

const PrintMode = enum {
    quiet,
    /// print loop intermediate results in run()
    repl,
};

pub fn init(buffer: []align(Region.Header.alignment) u8, w: *Io.Writer, print_mode: PrintMode) !Interp {
    var r: Region = try .initBuffer(buffer);
    var p: Interp = .{
        .nil = undefined,
        .tru = undefined,
        .env = undefined,
        .w = w,
        .print_mode = print_mode,
        .exprs = .{},
        .atoms = .{},
        .regions = .{},
        .root_checkpoint = undefined,
    };
    try p.regions.append(r.allocator(), r);

    p.nil = try p.dupeExprId(box(.nil, 0));
    p.tru = try p.atom("t");
    // trace("p.nil {} p.t {}", .{ p.nil.expr(&p).ord(), p.tru.expr(&p).ord() });
    p.env = try p.pair(p.tru, p.tru, p.nil);
    for (0..prims_len) |i| {
        const pr: Prim = @enumFromInt(i);
        // trace("prim {t}", .{pr});
        p.env = try p.pair(
            try p.atom(@tagName(pr)),
            try p.dupeExprId(box(.prim, @intCast(i))),
            p.env,
        );
    }
    p.root_checkpoint = p.rootRegion().checkpoint();
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

    pub const Id = enum(u48) {
        _,

        pub fn int(i: Id) u48 {
            return @intFromEnum(i);
        }

        pub fn expr(i: Id, p: *Interp) *Expr {
            return &p.exprs.items[@intFromEnum(i)];
        }

        pub fn fmt(e: Expr.Id, p: *Interp) Fmt {
            return .{ .e = e, .p = p };
        }

        pub fn iterator(e: Expr.Id, p: *Interp) Iterator {
            return .init(e, p);
        }

        pub fn equStructural(a: Expr.Id, b: Expr.Id, p: *Interp) bool {
            return switch (a.expr(p).boxed.tag.pair(b.expr(p).boxed.tag)) {
                Expr.Tag.pair(.atom, .atom) => std.mem.eql(u8, p.atomName(a), p.atomName(b)),
                Expr.Tag.pair(.nil, .nil) => true,
                Expr.Tag.pair(.cons, .cons) => p.carAssume(a).equStructural(p.carAssume(b), p) and
                    p.cdrAssume(a).equStructural(p.cdrAssume(b), p),
                else => a.expr(p).int == b.expr(p).int,
            };
        }
    };

    pub fn ord(x: *const Expr) u48 {
        return x.boxed.payload;
    }

    pub fn not(x: Expr) bool {
        return x.boxed.tag.int() == NIL_;
    }

    pub fn equ(x: Expr, y: Expr) bool {
        return x.int == y.int;
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
        e: Expr.Id,

        pub fn format(f: Fmt, w: *Io.Writer) !void {
            return f.formatFallible(w) catch return error.WriteFailed;
        }

        fn formatList(f: Fmt, w: *Io.Writer) !void {
            var t = f.e;
            while (true) {
                try (try f.p.car(t)).fmt(f.p).format(w);
                t = try f.p.cdr(t);
                switch (t.expr(f.p).boxed.tag.int()) {
                    NIL_ => break,
                    CONS => {},
                    else => {
                        try w.writeAll(" . ");
                        try t.fmt(f.p).format(w);
                        break;
                    },
                }
                try w.writeByte(' ');
            }
        }

        fn formatFallible(f: Fmt, w: *Io.Writer) Error!void {
            const e = f.e.expr(f.p).*;
            switch (e.boxed.tag.int()) {
                NIL_ => try w.writeAll("()"),
                ATOM => try w.writeAll(f.p.atomName(f.e)),
                PRIM => try w.print("<{t}>", .{@as(Prim, @enumFromInt(e.ord()))}),
                CONS => {
                    const is_not_root = 1 - e.boxed.sign; // sign set to 1 for top level list
                    if (f.p.isQuote(f.e)) {
                        try w.writeByte('\'');
                        try formatList(f.p.cdrAssume(f.e).fmt(f.p), w);
                    } else {
                        try w.writeAll("("[0..is_not_root]);
                        try formatList(f, w);
                        try w.writeAll(")"[0..is_not_root]);
                    }
                },
                CLOS => try w.print("{{{d}}}", .{e.ord()}),
                MACR => try w.print("[{d}]", .{e.ord()}),
                else => try w.print("{d}", .{e.float}),
            }
        }
    };

    pub fn equStructural(a: *Expr, b: *Expr, p: *Interp) bool {
        return a.id(p).equStructural(b.id(p), p);
    }

    pub fn id(e: *Expr, p: *Interp) Id {
        return p.exprId(e);
    }

    // TODO check tag
    /// convert or check number n
    pub fn toNum(e: Expr) Expr {
        return e;
    }
};

pub const ATOM = Expr.Tag.atom.int();
pub const PRIM = Expr.Tag.prim.int();
pub const CONS = Expr.Tag.cons.int();
pub const CLOS = Expr.Tag.clos.int();
pub const MACR = Expr.Tag.macr.int();
pub const NIL_ = Expr.Tag.nil.int();

pub fn box(t: Expr.Tag, payload: u48) Expr {
    return .{ .boxed = .{ .tag = t, .payload = payload } };
}

pub fn rootRegion(p: *Interp) *Region {
    return &p.regions.items[0];
}

// pub fn region(p: *Interp) *Region {
//     return &p.regions.items[p.regions.items.len - 1];
// }

pub fn atom(p: *Interp, s: []const u8) !Expr.Id {
    const id = p.atoms.count();
    const gop = try p.atoms.getOrPut(p.rootRegion().allocator(), s);
    if (!gop.found_existing) {
        const e = try p.dupeExprId(box(.atom, @intCast(id)));
        gop.value_ptr.* = e;
    }
    return gop.value_ptr.*;
}

/// unquotes quoted atoms and asserts they end with a quote
pub fn atomName(p: *Interp, e: Expr.Id) []const u8 {
    const ee = e.expr(p).*;
    assert(ee.boxed.tag == .atom);
    const ret = p.atoms.keys()[ee.ord()];
    const quoted = @intFromBool(ret[0] == '"');
    assert(quoted == 0 or ret[ret.len - 1] == '"');
    return ret[quoted .. ret.len - quoted];
}

fn createExpr(p: *Interp) !*Expr {
    return try p.exprs.addOne(p.rootRegion().allocator());
}

fn createExprs(p: *Interp, len: u16) ![]Expr {
    return p.exprs.addManyAsSlice(p.rootRegion().allocator(), len);
}

fn dupeExpr(p: *Interp, e: Expr) !*Expr {
    const ret = try p.createExpr();
    ret.* = e;
    return ret;
}

fn dupeExprId(p: *Interp, e: Expr) !Expr.Id {
    return (try p.dupeExpr(e)).id(p);
}

fn exprId(p: *Interp, e: *const Expr) Expr.Id {
    return @enumFromInt(e - p.exprs.items.ptr);
}

pub fn cons(p: *Interp, x: Expr.Id, y: Expr.Id) !Expr.Id {
    const es = try p.createExprs(3);
    @memcpy(es, &[_]Expr{ box(.cons, es[1].id(p).int()), x.expr(p).*, y.expr(p).* });
    return es[0].id(p);
}

pub fn consFromSlice(p: *Interp, exprs: []const Expr.Id) !Expr.Id {
    var i = exprs.len - 1;
    var c = try p.cons(exprs[i], p.nil);
    while (i > 0) {
        i -= 1;
        c = try p.cons(exprs[i], c);
    }
    return c;
}

pub fn pair(p: *Interp, v: Expr.Id, x: Expr.Id, e: Expr.Id) !Expr.Id {
    return try p.cons(try p.cons(v, x), e);
}

pub fn isQuote(p: *Interp, t: Expr.Id) bool {
    const a = p.carAssume(t);
    return a.expr(p).boxed.tag == .atom and
        std.mem.eql(u8, "quote", p.atomName(a));
}

/// parse src and then loop evaluating intermediates
pub fn run(p: *Interp, src: [:0]const u8, file_path: []const u8) Error!Expr.Id {
    var t = try p.parse(src, file_path);
    var result = p.nil;
    while (t.expr(p).boxed.tag == .cons) {
        result = try p.eval(p.carAssume(t), p.env);
        if (p.print_mode == .repl)
            try p.w.print("{f}\n", .{result.fmt(p)});
        t = p.cdrAssume(t);
    }
    if (p.print_mode == .repl) try p.w.flush();
    return result;
}

pub fn parse(p: *Interp, src: [:0]const u8, file_path: []const u8) Error!Expr.Id {
    // trace("parse() called", .{});
    var t = Tokenizer{
        .src = src,
        .pos = 0,
        .file_path = file_path,
    };

    var e = try p.parseList(&t, .eof);
    assert(e.expr(p).boxed.tag == .cons);
    e.expr(p).boxed.sign = 1; // sign=1 means top level list
    return e;
}

fn parseExpr(p: *Interp, t: *Tokenizer, token: Tokenizer.Token) Error!Expr.Id {
    return p.parseExprInner(t, token) catch |e|
        p.err(e, "{f}", .{token.fmt(t)});
}

fn parseExprInner(p: *Interp, t: *Tokenizer, token: Tokenizer.Token) Error!Expr.Id {
    // trace("parseExpr({f})", .{token.fmt(t)});
    const src = token.src(t.src);

    return switch (token.tag) {
        .lparen => try p.parseList(t, .rparen),
        .symbol, .string => if (std.mem.eql(u8, src, "nil"))
            p.nil
        else
            try p.atom(src),
        .number => try p.dupeExprId(.{ .float = try std.fmt.parseFloat(f64, src) }),
        .eof => p.nil,
        .quote => {
            const quoted = try p.parseExprInner(t, t.next());
            return p.cons(try p.atom("quote"), try p.cons(quoted, p.nil));
        },
        .rparen => return error.Parse,
        else => |tag| std.debug.panic("TODO handle {t}", .{tag}),
    };
}

fn parseList(p: *Interp, t: *Tokenizer, comptime end_tag: Tokenizer.Token.Tag) Error!Expr.Id {
    const token = t.next();
    // trace("parseList() called {f}", .{token.fmt(t)});
    switch (token.tag) {
        end_tag => return p.nil,
        .dot => {
            const last_expr = try p.parseExprInner(t, t.next());
            const right_paren = t.next();
            // trace("parseList() right_paren {f}", .{right_paren.fmt(t)});
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

pub fn gc(p: *Interp) void {
    p.rootRegion().restore(p.root_checkpoint);
}

pub fn carOpt(p: *Interp, x: Expr.Id) ?Expr.Id {
    const e = x.expr(p).*;
    return if (e.boxed.tag.isOneOf(&.{ CONS, CLOS, MACR }))
        @enumFromInt(e.ord())
    else
        null;
}

pub fn car(p: *Interp, x: Expr.Id) Error!Expr.Id {
    return p.carOpt(x) orelse p.err(error.NonPair, "{f}", .{x.fmt(p)});
}

// TODO replace car with carAssume whenever CONS/CLOS/MACR is known
pub fn carAssume(p: *Interp, x: Expr.Id) Expr.Id {
    return p.carOpt(x).?;
}

pub fn cdrOpt(p: *Interp, x: Expr.Id) ?Expr.Id {
    const e = x.expr(p).*;
    return if (e.boxed.tag.isOneOf(&.{ CONS, CLOS, MACR }))
        @enumFromInt(e.ord() + 1)
    else
        null;
}

pub fn cdr(p: *Interp, x: anytype) Error!Expr.Id {
    return p.cdrOpt(x) orelse p.err(error.NonPair, "{f}", .{x.fmt(p)});
}

pub fn cdrAssume(p: *Interp, x: Expr.Id) Expr.Id {
    return p.cdrOpt(x).?;
}

/// construct a closure, returns a boxed CLOS
fn closure(p: *Interp, vars: Expr.Id, body: Expr.Id, e: Expr.Id) Error!Expr.Id {
    const pr = try p.pair(vars, body, if (e.expr(p).equ(p.env.expr(p).*)) p.nil else e);
    return try p.dupeExprId(box(.clos, pr.expr(p).ord()));
}

fn macro(p: *Interp, v: Expr.Id, x: Expr.Id) Error!Expr.Id {
    return try p.dupeExprId(box(.macr, (try p.cons(v, x)).expr(p).ord()));
}

/// look up a symbol in an environment, return its value or ERR if not found
fn assoc(p: *Interp, a: Expr.Id, e0: Expr.Id) Error!Expr.Id {
    var e = e0;
    while (e.expr(p).boxed.tag.int() == CONS and
        !a.expr(p).equ((try p.car(p.carAssume(e))).expr(p).*))
    {
        e = p.cdrAssume(e);
    }

    return if (e.expr(p).boxed.tag.int() == CONS)
        try p.cdr(p.carAssume(e))
    else
        p.err(error.Unbound, "{f}", .{a.fmt(p)});
}

fn let(p: *Interp, x: Expr.Id) bool {
    return !x.expr(p).not() and !(if (p.cdrOpt(x)) |y| y.expr(p).not() else true);
}

const BindMode = enum { normal, optional, rest };

/// create environment by extending `env` with variables `vars` bound to values `vals`
/// `&rest` is a parameter pattern that allows a function to accept a variable
/// number of arguments, collecting all remaining arguments into a list.
/// `&optional` pattern marks an argument as optional with default value nil.
fn bind(p: *Interp, vars: Expr.Id, vals: Expr.Id, e: Expr.Id) Error!Expr.Id {
    return p.bindMode(vars, vals, e, .normal);
}
fn bindMode(p: *Interp, vars: Expr.Id, vals: Expr.Id, e: Expr.Id, mode: BindMode) Error!Expr.Id {
    trace("bind vars {f}", .{vars.fmt(p)});
    return switch (vars.expr(p).boxed.tag.int()) {
        NIL_ => e,
        CONS => blk: {
            const v = try p.car(vars);
            trace("bind v {f}", .{v.fmt(p)});
            if (v.expr(p).boxed.tag == .atom) {
                const name = p.atomName(v);
                if (std.mem.eql(u8, "&optional", name)) {
                    break :blk try p.bindMode(try p.cdr(vars), vals, e, .optional);
                }
                if (std.mem.eql(u8, "&rest", name)) { // wrap rest args in list
                    const rest = try p.car(try p.cdr(vars));
                    break :blk p.pair(rest, vals, e);
                }
            }

            break :blk switch (mode) {
                .normal => p.bindMode(
                    try p.cdr(vars),
                    try p.cdr(vals),
                    try p.pair(v, try p.car(vals), e),
                    .normal,
                ),
                .optional => if (vals.expr(p).boxed.tag.int() == CONS)
                    p.bindMode(
                        try p.cdr(vars),
                        try p.cdr(vals),
                        try p.pair(v, try p.car(vals), e),
                        .optional,
                    )
                else
                    p.bindMode(
                        try p.cdr(vars),
                        vals,
                        try p.pair(v, p.nil, e),
                        .optional,
                    ),
                .rest => unreachable,
            };
        },
        else => p.pair(vars, vals, e),
    };
}

/// return a new list of evaluated Lisp expressions t in environment e
fn evlis(p: *Interp, t: Expr.Id, e: Expr.Id) Error!Expr.Id {
    return switch (t.expr(p).boxed.tag.int()) {
        CONS => p.cons(
            try p.eval(p.carAssume(t), e),
            try p.evlis(p.cdrAssume(t), e),
        ),
        ATOM => p.assoc(t, e),
        else => p.nil,
    };
}

/// apply closure `clos` to arguments `args` in environment `e`
pub fn reduce(p: *Interp, clos: Expr.Id, args: Expr.Id, e: Expr.Id) Error!Expr.Id {
    const clos_fun = try p.car(clos);
    const clos_env = try p.cdr(clos);
    const clos_vars = try p.car(clos_fun);
    const clos_body = try p.cdr(clos_fun);
    const eval_args = try p.evlis(args, e);
    return p.eval(clos_body, try p.bind(
        clos_vars,
        eval_args,
        if (clos_env.expr(p).not()) p.env else clos_env,
    ));
}

fn callPrim(pm: Prim, p: *Interp, t: Expr.Id, e: Expr.Id) Error!Expr.Id {
    // trace("callPrim({t}, {f})", .{ pm, t.fmt(p) });
    return switch (pm) {
        inline else => |tag| try @field(primitives, @tagName(tag))(p, t, e),
    };
}

fn doApply(p: *Interp, f: Expr.Id, t: Expr.Id, e: Expr.Id) Error!Expr.Id {
    trace("doApply {f} {f}", .{ f.fmt(p), t.fmt(p) });
    return switch (f.expr(p).boxed.tag.int()) {
        PRIM => try callPrim(@enumFromInt(f.expr(p).ord()), p, t, e),
        CLOS => try p.reduce(f, t, e),
        MACR => blk: {
            var d = e;
            var x = t;
            var v = try p.car(f);
            while (v.expr(p).boxed.tag == .cons) {
                d = try p.pair(try p.car(v), try p.car(x), d);
                trace("doApply macr x {f} v {f}", .{ x.fmt(p), v.fmt(p) });
                x = try p.cdr(x);
                v = try p.cdr(v);
            }
            if (v.expr(p).boxed.tag == .atom) d = try p.pair(v, x, d);
            const expanded = try p.eval(try p.cdr(f), d);
            break :blk try p.eval(expanded, e);
        },
        else => p.err(error.CannotApply, "{s} {f}", .{ f.expr(p).tagName(), f.fmt(p) }),
    };
}

fn eval(p: *Interp, t: Expr.Id, e: Expr.Id) Error!Expr.Id {
    errdefer trace("eval error on {f}", .{t.fmt(p)});
    const ret = switch (t.expr(p).boxed.tag.int()) {
        ATOM => try p.assoc(t, e),
        CONS => try p.doApply(try p.eval(p.carAssume(t), e), p.cdrAssume(t), e),
        MACR => {
            // bind macro f variables v to the given arguments literally (i.e.
            // without evaluating the arguments)
            var d = e;
            const f = try p.eval(try p.car(t), e);
            var v = try p.car(f);
            var x = try p.cdr(t);
            while (v.expr(p).boxed.tag.int() == CONS) {
                d = try p.pair(try p.car(v), try p.car(x), d);
                v = try p.cdr(v);
                x = try p.cdr(x);
            }
            if (v.expr(p).boxed.tag.int() == ATOM) {
                d = try p.pair(v, x, d);
            }
            // expand macro f, then continue evaluating the expanded x
            return try p.eval(try p.cdr(f), d);
        },
        else => t,
    };
    trace("eval\n  {f}\n  => {f}", .{ t.fmt(p), ret.fmt(p) });
    return ret;
}

pub fn fromBool(p: *Interp, b: bool) Expr.Id {
    return if (b) p.tru else p.nil;
}

// TODO unify with Expr.Fmt
pub const Iterator = struct {
    cur: Expr.Id,
    p: *Interp,

    pub fn init(cur: Expr.Id, p: *Interp) Iterator {
        return .{ .cur = cur, .p = p };
    }

    pub fn next(i: *Iterator) ?Expr.Id {
        const ret = i.p.carOpt(i.cur) orelse return null;
        i.cur = i.p.cdrAssume(i.cur);
        return ret;
    }
};

pub const Prim = std.meta.DeclEnum(primitives);
pub const prims_len = @typeInfo(Prim).@"enum".fields.len;

// TODO maybe accept `p: *primitives` instead of *Interp and do
// @fieldParentPtr mixins
const primitives = struct {
    /// (eval x) return evaluated x (such as when x was quoted)
    pub fn eval(p: *Interp, t: Expr.Id, e: Expr.Id) Error!Expr.Id {
        return try p.eval(try p.car(try p.evlis(t, e)), e);
    }
    /// (quote x) special form, returns x unevaluated "as is"
    pub fn quote(p: *Interp, t: Expr.Id, _: Expr.Id) Error!Expr.Id {
        if (!(try p.cdr(t)).expr(p).not())
            return p.err(error.User, "quote expected one arg. found {f}", .{t.fmt(p)});
        return try p.car(t);
    }
    /// (cons x y) construct pair (x . y)
    pub fn cons(p: *Interp, t: Expr.Id, e: Expr.Id) Error!Expr.Id {
        const t1 = try p.evlis(t, e);
        return p.cons(try p.car(t1), try p.car(try p.cdr(t1)));
    }
    /// (car p) car of pair p
    pub fn car(p: *Interp, t: Expr.Id, e: Expr.Id) Error!Expr.Id {
        return try p.car(try p.car(try p.evlis(t, e)));
    }
    /// (cdr p) cdr of pair p
    pub fn cdr(p: *Interp, t: Expr.Id, e: Expr.Id) Error!Expr.Id {
        return try p.cdr(try p.car(try p.evlis(t, e)));
    }
    // (list x1 x2 ... xk)
    // returns the list of `x1`, `x2`, ..., `xk`.  That is, `(x1 x2 ... xk)` with all `x` evaluated.
    pub fn list(p: *Interp, t: Expr.Id, e: Expr.Id) Error!Expr.Id {
        return p.evlis(t, e);
    }
    /// (apply f arg1 arg2 ... argn-list)
    /// argn-list is a list
    pub fn apply(p: *Interp, t: Expr.Id, e: Expr.Id) Error!Expr.Id {
        if (t.expr(p).not())
            return p.err(error.User, "apply missing function arg", .{});

        var f = try p.car(t);
        if (f.expr(p).boxed.tag == .cons and p.isQuote(f))
            f = p.carAssume(p.cdrAssume(f));

        f = try p.eval(f, e);
        if (f.expr(p).boxed.tag == .atom)
            f = try p.assoc(f, p.env);

        const args = try p.evlis(try p.cdr(t), e); // args = (arg1 arg2 ... argn list)
        if (args.expr(p).not()) return try p.doApply(f, args, e);

        // build argument list
        var buf: [64]Expr.Id = undefined;
        var argsb: std.ArrayList(Expr.Id) = .initBuffer(&buf);
        var current = args;
        while (current.expr(p).boxed.tag == .cons) {
            const next = p.cdrAssume(current);
            // trace("apply current:{f} next:{f}", .{ current.fmt(p), next.fmt(p) });
            if (next.expr(p).boxed.tag != .cons) {
                const list_part = try p.car(current);
                // trace("apply list_part:{f} current:{f}", .{ list_part.fmt(p), current.fmt(p) });
                if (list_part.expr(p).boxed.tag == .cons) {
                    var iter = list_part.iterator(p);
                    while (iter.next()) |x| try argsb.appendBounded(x);
                } else try argsb.appendBounded(list_part);
                break;
            }
            try argsb.appendBounded(try p.car(current));
            current = next;
        }
        const new_args = try p.consFromSlice(argsb.items);
        // trace("apply new_args {f}", .{new_args.fmt(p)});
        return try p.doApply(f, new_args, e);
    }
    /// (funcall function args)
    /// applies function to args. If function is a symbol, it is coerced to a
    /// function as if by finding its functional value
    pub fn funcall(p: *Interp, t: Expr.Id, e: Expr.Id) Error!Expr.Id {
        if (t.expr(p).not()) return p.err(error.User, "funcall missing function arg", .{});
        var f = try p.car(t);
        if (f.expr(p).boxed.tag == .cons and p.isQuote(f))
            f = p.carAssume(p.cdrAssume(f));
        f = try p.eval(f, e);
        if (f.expr(p).boxed.tag == .atom)
            f = try p.assoc(f, p.env);
        // Evaluate all arguments normally (no list splicing like apply)
        const args = try p.evlis(try p.cdr(t), e);
        return try p.doApply(f, args, e);
    }
    const Op = enum { add, sub, mul, div };
    fn mathOp(p: *Interp, t: Expr.Id, e: Expr.Id, comptime op: Op) Error!Expr.Id {
        var t1 = try p.evlis(t, e);
        var n = (try p.car(t1)).expr(p).*;
        while (true) {
            t1 = try p.cdr(t1);
            if (t1.expr(p).not()) break;
            switch (op) {
                .add => n.float += (try p.car(t1)).expr(p).toNum().float,
                .sub => n.float -= (try p.car(t1)).expr(p).toNum().float,
                .mul => n.float *= (try p.car(t1)).expr(p).toNum().float,
                .div => n.float /= (try p.car(t1)).expr(p).toNum().float,
            }
        }
        return p.dupeExprId(n.toNum());
    }
    /// (+ n1 n2 ... nk) sum of n1 to nk
    pub fn @"+"(p: *Interp, t: Expr.Id, e: Expr.Id) Error!Expr.Id {
        return mathOp(p, t, e, .add);
    }
    /// (- n1 n2 ... nk) sum of n1 to nk
    pub fn @"-"(p: *Interp, t: Expr.Id, e: Expr.Id) Error!Expr.Id {
        return mathOp(p, t, e, .sub);
    }
    /// (* n1 n2 ... nk) sum of n1 to nk
    pub fn @"*"(p: *Interp, t: Expr.Id, e: Expr.Id) Error!Expr.Id {
        return mathOp(p, t, e, .mul);
    }
    /// (/ n1 n2 ... nk) sum of n1 to nk
    pub fn @"/"(p: *Interp, t: Expr.Id, e: Expr.Id) Error!Expr.Id {
        return mathOp(p, t, e, .div);
    }
    /// (truncate n) integer part of n, round toward 0
    pub fn truncate(p: *Interp, t: Expr.Id, e: Expr.Id) Error!Expr.Id {
        const n = (try p.car(try p.evlis(t, e)));
        // @floatFromInt(@as(u128, @intFromFloat(n.float)))
        return try p.dupeExprId(.{ .float = @trunc(n.expr(p).float) });
    }
    fn cmpOp(p: *Interp, t: Expr.Id, e: Expr.Id, comptime cmp: enum { lt, gt, eq, le, ge }) Error!Expr.Id {
        const t1 = try p.evlis(t, e);
        const x = (try p.car(t1)).expr(p);
        const y = (try p.car(try p.cdr(t1))).expr(p);
        return p.fromBool(switch (cmp) {
            .lt => x.float - y.float < 0,
            .gt => x.float - y.float > 0,
            .eq => x.equ(y.*),
            .le => x.float - y.float <= 0,
            .ge => x.float - y.float >= 0,
        });
    }
    /// (< n1 n2) t if n1<n2, otherwise ()
    pub fn @"<"(p: *Interp, t: Expr.Id, e: Expr.Id) Error!Expr.Id {
        return cmpOp(p, t, e, .lt);
    }
    /// (> n1 n2) t if n1>n2, otherwise ()
    pub fn @">"(p: *Interp, t: Expr.Id, e: Expr.Id) Error!Expr.Id {
        return cmpOp(p, t, e, .gt);
    }
    /// (= n1 n2) t if n1=n2, otherwise ()
    pub fn @"="(p: *Interp, t: Expr.Id, e: Expr.Id) Error!Expr.Id {
        return cmpOp(p, t, e, .eq);
    }
    /// (<= n1 n2) t if n1<=n2, otherwise ()
    pub fn @"<="(p: *Interp, t: Expr.Id, e: Expr.Id) Error!Expr.Id {
        return cmpOp(p, t, e, .le);
    }
    /// (>= n1 n2) t if n1>=n2, otherwise ()
    pub fn @">="(p: *Interp, t: Expr.Id, e: Expr.Id) Error!Expr.Id {
        return cmpOp(p, t, e, .ge);
    }
    /// (or x1 x2 ... xk) first x that is truthy, otherwise ()
    pub fn @"or"(p: *Interp, t: Expr.Id, e: Expr.Id) Error!Expr.Id {
        var x = p.nil;
        var t1 = t;
        while (true) {
            if (t1.expr(p).not()) break;
            x = try p.eval(try p.car(t1), e);
            if (!x.expr(p).not()) break;
            t1 = try p.cdr(t1);
        }
        return x;
    }
    /// (and x1 x2 ... xk) last x if all x are truthy, otherwise ()
    pub fn @"and"(p: *Interp, t: Expr.Id, e: Expr.Id) Error!Expr.Id {
        var x = p.nil;
        var t1 = t;
        while (true) {
            if (t1.expr(p).not()) break;
            x = try p.eval(try p.car(t1), e);
            if (x.expr(p).not()) break;
            t1 = try p.cdr(t1);
        }
        return x;
    }
    /// (if x y z) if x then y else z
    pub fn @"if"(p: *Interp, t: Expr.Id, e: Expr.Id) Error!Expr.Id {
        const cnd = try p.eval(try p.car(t), e);
        const branches = try p.cdr(t);
        const res = try p.car(if (cnd.expr(p).not())
            try p.cdr(branches)
        else
            branches);
        // trace("if cnd {f} branches {f} res {f}", .{ cnd.fmt(p), branches.fmt(p), res.fmt(p) });
        return try p.eval(res, e);
    }
    /// (defvar v x) define a named value globally
    pub fn defvar(p: *Interp, t: Expr.Id, e: Expr.Id) Error!Expr.Id {
        const t1 = try p.cdr(t);
        const x = try p.car(t1);
        const t2 = try p.eval(x, e);
        // trace("fst {f} t1 {f} x {f} t2 {f}", .{ (try p.car(t)).fmt(p), t1.fmt(p), x.fmt(p), t2.fmt(p) });
        p.env = try p.pair(try p.car(t), t2, p.env);
        return try p.car(t);
    }
    // (defun name params body) define a named function globally
    pub fn defun(p: *Interp, t: Expr.Id, e: Expr.Id) Error!Expr.Id {
        const name = try p.car(t);
        const params = try p.car(try p.cdr(t));
        const body = try p.car(try p.cdr(try p.cdr(t)));
        trace("defun name:{f} params:{f} body:{f}", .{ name.fmt(p), params.fmt(p), body.fmt(p) });
        p.env = try p.pair(name, try p.closure(params, body, e), p.env);
        return name;
    }
    pub fn cond(p: *Interp, t0: Expr.Id, e: Expr.Id) Error!Expr.Id {
        var t = t0;
        while ((try p.eval(try p.car(try p.car(t)), e)).expr(p).not()) {
            t = try p.cdr(t);
        }
        return try p.car(try p.cdr(try p.car(t)));
    }
    /// (not x) t if x is (), otherwise ()
    pub fn not(p: *Interp, t: Expr.Id, e: Expr.Id) Error!Expr.Id {
        return p.fromBool((try p.car(try p.evlis(t, e))).expr(p).not());
    }
    /// (lambda v x) construct a closure
    pub fn lambda(p: *Interp, t: Expr.Id, e: Expr.Id) Error!Expr.Id {
        return p.closure(try p.car(t), try p.car(try p.cdr(t)), e);
    }
    /// (let* ((k1 v1) (k2 v2) ... (kk vk)) y1 ... yn)
    /// eval `y`s with `kv`s in e0
    pub fn @"let*"(p: *Interp, t: Expr.Id, e0: Expr.Id) Error!Expr.Id {
        var kvs = try p.car(t);
        var kv = try p.car(kvs);
        var e = e0;
        while (p.let(kv)) {
            e = try p.pair(
                try p.car(kv),
                try p.eval(try p.car(try p.cdr(kv)), e),
                e,
            );
            kvs = p.cdrOpt(kvs) orelse break;
            kv = p.carOpt(kvs) orelse break;
        }

        var current = p.cdrOpt(t) orelse return p.nil;
        var result = p.nil;
        while (current.expr(p).boxed.tag.int() == CONS) {
            result = try p.eval(p.carAssume(current), e);
            current = p.cdrAssume(current);
        }
        return result;
    }
    pub fn consp(p: *Interp, t: Expr.Id, e: Expr.Id) Error!Expr.Id {
        const x = try p.car(try p.evlis(t, e));
        return p.fromBool(x.expr(p).boxed.tag == .cons);
    }
    /// (progn (x1 x2 ...)) evaluates each x$i, returning the last one
    pub fn progn(p: *Interp, t: Expr.Id, e: Expr.Id) Error!Expr.Id {
        var result = p.nil;
        var current = t;
        while (current.expr(p).boxed.tag == .cons) {
            result = try p.eval(try p.car(current), e);
            current = try p.cdr(current);
        }
        return result;
    }
    pub fn print(p: *Interp, t: Expr.Id, e: Expr.Id) Error!Expr.Id {
        const t1 = try p.car(try p.evlis(t, e));
        try p.w.print("{f}\n", .{t1.fmt(p)});
        try p.w.flush();
        return t1;
    }
    pub fn load(p: *Interp, t: Expr.Id, _: Expr.Id) Error!Expr.Id {
        const t1 = try p.car(t);
        if (t1.expr(p).boxed.tag != .atom)
            return p.err(error.User, "expected atom. found {f}", .{t1.fmt(p)});
        const name = p.atomName(t1);
        trace("load name {s}\n", .{name});
        const f = try std.fs.cwd().openFile(name, .{});
        defer f.close();
        const len = try f.getEndPos();
        const mem = try p.rootRegion().allocator().allocSentinel(u8, len, 0);
        const amt = try f.read(mem);
        assert(amt == len);
        mem[len] = 0;
        return try p.run(mem[0..len :0], name);
    }
    pub fn macro(p: *Interp, t: Expr.Id, _: Expr.Id) Error!Expr.Id {
        return p.macro(try p.car(t), try p.car(try p.cdr(t)));
    }
    /// object equality - whether two objects are the same object in memory
    pub fn eq(p: *Interp, t: Expr.Id, e: Expr.Id) Error!Expr.Id {
        const a = (try p.eval(try p.car(t), e));
        const b = (try p.eval(try p.car(try p.cdr(t)), e));
        return p.fromBool(if (a.expr(p).isTagged() and b.expr(p).isTagged())
            a.expr(p).ord() == b.expr(p).ord()
        else
            a.expr(p).int == b.expr(p).int);
    }
    /// structural equality
    pub fn equal(p: *Interp, t: Expr.Id, e: Expr.Id) Error!Expr.Id {
        const a = try p.eval(try p.car(t), e);
        const b = try p.eval(try p.car(try p.cdr(t)), e);
        return p.fromBool(a.expr(p).equStructural(b.expr(p), p));
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
const tests = @import("tests.zig");
const testEval = tests.testEval;
const Region = @import("Region");

test {
    _ = primitives;
    _ = tests;
}
