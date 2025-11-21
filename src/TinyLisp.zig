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
    /// print loop intermediate results
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
        macr = 0x7ffc, //  0b0111111111111_100
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

    /// convert or check number n (does nothing, e.g. could check for NaN)
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

pub fn gc(p: *Interp, x: Expr) Expr {
    _ = p;
    // TODO implement
    return x;
}

pub fn rc(p: *Interp, x: *Expr, y: Expr) Expr {
    _ = p;
    _ = x;
    // TODO implement
    return y;
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

// create environment by extending `env` with variables `vars` bound to values `vals` */
fn bind(p: *Interp, vars: Expr, vals: Expr, e: Expr) Error!Expr {
    return switch (vars.boxed.tag.int()) {
        NIL_ => e,
        CONS => p.bind(
            try p.cdr(vars),
            try p.cdr(vals),
            p.pair(try p.car(vars), try p.car(vals), e),
        ),
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

/// apply closure `clos` to arguments `args` in environment `env`
pub fn reduce(l: *Interp, clos: Expr, args: Expr, e: Expr) Error!Expr {
    const clos_fun = try l.car(clos);
    const clos_env = try l.cdr(clos);
    const clos_vars = try l.car(clos_fun);
    const clos_body = try l.cdr(clos_fun);
    const eval_args = try l.evlis(args, e);
    return l.eval(clos_body, try l.bind(
        clos_vars,
        eval_args,
        if (clos_env.not()) l.env else clos_env,
    ));
}

fn callPrim(pm: Prim, p: *Interp, t: Expr, e: Expr) Error!Expr {
    // trace("callPrim({t}, {f})", .{ pm, t.fmt(p) });
    return switch (pm) {
        inline else => |tag| try @field(primitives, @tagName(tag))(p, t, e),
    };
}

fn doApply(p: *Interp, f: Expr, t: Expr, e: Expr) Error!Expr {
    trace("apply {f} {f}", .{ f.fmt(p), t.fmt(p) });
    return switch (f.boxed.tag.int()) {
        PRIM => try callPrim(@enumFromInt(f.ord()), p, t, e),
        CLOS => try p.reduce(f, t, e),
        MACR => unreachable,
        else => |i| if (i == CONS and p.isQuote(f))
            try p.doApply(try p.assoc(p.carAssume(p.cdrAssume(f)), e), t, e)
        else
            p.err(error.CannotApply, "{s} {f}", .{ f.tagName(), f.fmt(p) }),
    };
}

fn dup(_: *Interp, x: Expr) Expr {
    // TODO implement
    return x;
}

fn evarg(p: *Interp, t: *Expr, e: *Expr, a: *bool) Error!Expr {
    if (t.boxed.tag.int() == ATOM) {
        t.* = try p.assoc(t.*, e.*);
        a.* = true;
    }
    const x = try p.car(t.*);
    t.* = try p.cdr(t.*);
    return if (a.*) p.dup(x) else p.eval(x, e.*);
}

fn eval(p: *Interp, x0: Expr, e0: Expr) Error!Expr {
    var x = x0;
    var y = x;
    var e = e0;
    var d = nil;
    var f = nil;
    var g = nil;
    var h = nil;
    while (true) {
        trace("eval {s}: {f}", .{ x.tagName(), x.fmt(p) });
        // copy x to y to output y => x when tracing is enabled
        y = x;
        // if x is an atom, then return its value; if x is not an application
        // list (it is constant), then return x
        if (x.boxed.tag.int() == ATOM) {
            x = p.dup(try p.assoc(x, e));
            break;
        }
        if (x.boxed.tag.int() != CONS) {
            x = p.dup(x);
            break;
        }
        // save g = old f to garbage collect, evaluate f in the application (f . x) and get the list of arguments x
        g = f;
        f = try p.eval(try p.car(x), e);
        x = try p.cdr(x);
        if (f.boxed.tag.int() == PRIM) {
            // apply Lisp primitive to argument list x, return value in x
            // x = prim[ord(f)].f(x,&e);
            const pm: Prim = @enumFromInt(f.ord());
            // try p.w.print("pm {t}\n", .{pm});
            x = try callPrim(pm, p, x, e);
            // garbage collect g = old f, garbage collect old macro body h
            // gc(g);
            g = nil;
            //gc(h);
            h = nil;
            // if tail-call then continue evaluating x, otherwise return x
            if (isPrimTailCall(pm)) continue;
            break;
        }
        if (f.boxed.tag.int() == MACR) {
            // bind macro f variables v to the given arguments literally (i.e. without evaluating the arguments)
            d = p.dup(p.env);
            var v = try p.car(f);
            while (v.boxed.tag.int() == CONS) {
                d = p.pair(try p.car(v), p.dup(try p.car(x)), d);
                v = try p.cdr(v);
                x = try p.cdr(x);
            }
            if (v.boxed.tag.int() == ATOM) d = p.pair(v, p.dup(x), d);
            // expand macro f, then continue evaluating the expanded x
            x = try p.eval(try p.cdr(f), d);
            // garbage collect bindings d, gabage collect g = old f and old macro body h, save macro body h = x to gc later
            // gc(d);
            d = nil; // gc(g);
            g = nil; // gc(h);
            h = x;
            continue;
        }
        if (f.boxed.tag.int() == CONS) {}
        if (f.boxed.tag.int() != CLOS)
            return p.err(error.OutOfMemory, "{s} {f}", .{ f.tagName(), f.fmt(p) });
        // get the list of variables v of closure f and its local environment d (use global env when nil)
        var v = try p.car(try p.car(f));
        d = p.dup(try p.cdr(f));
        if (d.boxed.tag.int() == NIL_) d = p.dup(p.env);
        // bind closure f variables v to the evaluated argument values
        var a = false;
        while (v.boxed.tag.int() == CONS) {
            d = p.pair(p.carAssume(v), try p.evarg(&x, &e, &a), d);
            v = p.cdrAssume(v);
        }
        if (v.boxed.tag.int() == ATOM)
            d = p.pair(v, if (a) p.dup(x) else try p.evlis(x, e), d);
        // next, evaluate body x of closure f in environment e = d while keeping f in memory as long as x
        x = try p.cdr(try p.car(f));
        // discard copy of the old environment e to use new environment d
        // gc(e);
        e = d;
        d = nil;
        // garbage collect closure g = old f with old body x, garbage collect old macro body h
        // gc(g);
        g = nil;
        // gc(h);
        h = nil;
        // if (tr) trace(y, x, e);
        trace("eval {f} => {f}", .{ y.fmt(p), x.fmt(p) });
    }
    trace("eval {f} => {f}", .{ y.fmt(p), x.fmt(p) });

    // garbage collect environment e, closure f, macro body h
    // gc(e); gc(f); gc(h);
    // deregister 5 variables, if registered, without gc'ing them
    // rr(5);
    return x;
}

fn eval2(p: *Interp, x: Expr, e: Expr) Error!Expr {
    trace("eval({f})", .{x.fmt(p)});
    return switch (x.boxed.tag.int()) {
        ATOM => try p.assoc(x, e),
        CONS => try p.apply(
            try p.eval(try p.car(x), e),
            try p.cdr(x),
            e,
        ),
        MACR => {
            // bind macro f variables v to the given arguments literally (i.e.
            // without evaluating the arguments)
            var d = e;
            const f = try p.eval(try p.car(x), e);
            var v = try p.car(f);
            var xx = try p.cdr(x);
            while (v.boxed.tag.int() == CONS) {
                d = p.pair(try p.car(v), try p.car(xx), d);
                v = try p.cdr(v);
                xx = try p.cdr(xx);
            }
            if (v.boxed.tag.int() == ATOM) {
                d = p.pair(v, xx, d);
            }
            // expand macro f, then continue evaluating the expanded x
            return try p.eval(try p.cdr(f), d);
        },
        else => x,
    };
}

// TODO make streaming. replace src field with buffer and reader of some kind.  must be non seekable.
pub const Tokenizer = struct {
    src: [:0]const u8,
    file_path: []const u8,
    pos: u32 = 0,

    pub const Token = struct {
        start: u32,
        end: u32,
        tag: Tag,

        pub const Tag = enum(u8) {
            invalid,
            eof,
            lparen,
            rparen,
            symbol,
            number,
            string,
            quote,
            dot,
        };

        pub const Fmt = struct {
            t: Token,
            tzr: *Tokenizer,

            pub fn format(f: Fmt, w: *Io.Writer) !void {
                try w.print("{s}:{}:{} '{s}' {t}", .{
                    f.tzr.file_path,
                    f.t.start,
                    f.t.end,
                    f.tzr.src[f.t.start..f.t.end],
                    f.t.tag,
                });
            }
        };

        pub fn fmt(t: Token, tzr: *Tokenizer) Fmt {
            return .{ .t = t, .tzr = tzr };
        }

        pub fn src(t: Token, s: [:0]const u8) []const u8 {
            return s[t.start..t.end];
        }
    };

    const State = enum {
        init,
        symbol,
        whitespace,
        comment,
        number,
        number_dot,
        string,
        string_escape,
    };

    fn nextByte(t: *Tokenizer) u8 {
        defer t.pos += @intFromBool(t.pos <= t.src.len);
        return t.peek();
    }

    fn peek(t: *Tokenizer, n: u32) u8 {
        return t.src[t.pos + n];
    }

    fn eof(t: *Tokenizer) bool {
        return t.peek(0) == 0;
    }

    fn isSym(byte: u8) bool {
        return byte != ')' and byte != '(' and
            !std.ascii.isDigit(byte) and
            !isWhitespace(byte) and
            std.ascii.isPrint(byte);
    }

    fn isSym2(byte: u8) bool {
        return byte != ')' and byte != '(' and
            !isWhitespace(byte) and
            std.ascii.isPrint(byte);
    }

    fn isWhitespace(byte: u8) bool {
        // TODO benchmark which is faster
        if (false)
            return switch (byte) {
                ' ', '\n', '\t', '\r' => true,
                else => false,
            };

        const V = @Vector(4, u8);
        const x: [4]u8 = " \n\t\r".*;
        return @reduce(.Or, x == @as(V, @splat(byte)));
    }

    fn advance(self: *Tokenizer, n: u32) void {
        self.pos += n * @intFromBool(self.src[self.pos] != 0);
    }

    fn advanceTo(self: *Tokenizer, tag: @Type(.enum_literal), n: u32, t: *Token) State {
        self.advance(n);
        if (@hasField(Token.Tag, @tagName(tag)) and tag != .number) t.tag = tag;
        return switch (tag) {
            .lparen,
            .rparen,
            .invalid,
            .quote,
            .dot,
            // .quasi_quote,
            // .quasi_unquote,
            // .quasi_unquote_splicing,
            => .init,
            else => |ttag| @field(State, @tagName(ttag)),
        };
    }

    pub fn next(self: *Tokenizer) Token {
        var t: Token = undefined;

        state: switch (State.init) {
            inline else => |state| {
                const byte = self.peek(0);
                // trace("  state {t} byte '{c}'", .{ state, byte });
                switch (state) {
                    .init => {
                        t = .{ .tag = .invalid, .start = self.pos, .end = self.pos };
                        switch (byte) {
                            0 => t.tag = .eof,
                            '(' => _ = self.advanceTo(.lparen, 1, &t),
                            ')' => _ = self.advanceTo(.rparen, 1, &t),
                            '\'' => _ = self.advanceTo(.quote, 1, &t),
                            '.' => _ = self.advanceTo(.dot, 1, &t),
                            ';' => continue :state self.advanceTo(.comment, 1, &t),
                            '0'...'9' => continue :state self.advanceTo(.number, 1, &t),
                            '"' => continue :state self.advanceTo(.string, 1, &t),
                            '-', '+' => {
                                switch (self.peek(1)) {
                                    '1'...'9' => continue :state self.advanceTo(.number, 1, &t),
                                    '0' => self.pos += 2,
                                    else => continue :state self.advanceTo(.symbol, 1, &t),
                                }
                            },
                            else => if (isSym(byte)) {
                                continue :state self.advanceTo(.symbol, 1, &t);
                            } else if (isWhitespace(byte)) {
                                continue :state self.advanceTo(.whitespace, 1, &t);
                            } else {
                                std.debug.panic("unexpected byte '{c}'", .{byte});
                            },
                        }
                    },
                    .whitespace => if (isWhitespace(byte)) {
                        continue :state self.advanceTo(.whitespace, 1, &t);
                    } else {
                        continue :state .init;
                    },

                    .comment => {
                        self.advance(1);
                        switch (byte) {
                            '\n' => {
                                if (self.peek(0) == ';') {
                                    self.advance(1);
                                    continue :state .comment;
                                }
                            },
                            0 => {},
                            else => continue :state .comment,
                        }
                        continue :state .init;
                    },
                    .symbol => if (isSym2(byte)) {
                        continue :state self.advanceTo(.symbol, 1, &t);
                    },
                    .number => if (std.ascii.isDigit(byte)) {
                        continue :state self.advanceTo(.number, 1, &t);
                    } else if (byte == '.') {
                        continue :state self.advanceTo(.number_dot, 1, &t);
                    } else {
                        t.tag = .number;
                    },
                    .number_dot => if (std.ascii.isDigit(byte)) {
                        continue :state self.advanceTo(.number_dot, 1, &t);
                    } else {
                        t.tag = .number;
                    },
                    .string => switch (byte) {
                        '"' => self.advance(1),
                        '\\' => continue :state self.advanceTo(.string_escape, 0, &t),
                        else => continue :state self.advanceTo(.string, 1, &t),
                    },
                    .string_escape => {
                        var offset: usize = self.pos;
                        const r = std.zig.string_literal.parseEscapeSequence(self.src, &offset);
                        if (r == .success) {
                            continue :state self.advanceTo(.string, @intCast(offset - self.pos), &t);
                        } else {
                            _ = self.advanceTo(.invalid, @intCast(self.pos - offset), &t);
                        }
                    },
                }
            },
        }
        t.end = self.pos;
        // trace("{f}", .{t.fmt(self)});
        return t;
    }
};

pub fn fromBool(p: *Interp, b: bool) Expr {
    return if (b) p.tru else nil;
}

pub const Prim = std.meta.DeclEnum(primitives);
pub const prims_len = @typeInfo(Prim).@"enum".fields.len;

fn isPrimTailCall(p: Prim) bool {
    return switch (p) {
        .cond,
        .@"if",
        .@"let*",
        // .let,
        // .@"letrec*",
        // .letrec,
        => true,
        else => false,
    };
}

// TODO maybe accept `p: *primitives` instead of *Interp and do
// @fieldParentPtr mixins
const primitives = struct {
    /// (eval x) return evaluated x (such as when x was quoted)
    pub fn eval(p: *Interp, t: Expr, e: Expr) Error!Expr {
        return try p.eval(try p.car(try p.evlis(t, e)), e);
    }
    /// (quote x) special form, returns x unevaluated "as is"
    pub fn quote(p: *Interp, t: Expr, _: Expr) Error!Expr {
        return p.dup(try p.car(t));
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
    pub fn list(_: *Interp, t: Expr, _: Expr) Error!Expr {
        return t;
    }
    pub fn apply(p: *Interp, t: Expr, e: Expr) Error!Expr {
        return try p.doApply(p.carAssume(t), try p.car(p.cdrAssume(t)), e);
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
        const cnd = p.gc(try p.eval(try p.car(t), e));
        const branch = if (cnd.not()) try p.cdr(t) else t;
        return try p.car(try p.cdr(branch));
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
    /// eval `y`s with `kv`s in env
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
        // L f_macro(L t,L *_) { return macro(dup(car(t)),dup(car(cdr(t)))); }
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
    }
    test load {
        try testEval("!=", "(load src/common.lisp)");
    }
    test list {
        try testEval("(1 2)", "(list 1 2)");
    }
    test "macro" {
        try testEval("4", "(defun double (x) (+ x x)) (double 2)");
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
    try testParseFmt(.{ .file_path = "examples/basic.scm" });
    try testParseFmt(.{ .file_path = "examples/fizzbuzz.scm" });
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
