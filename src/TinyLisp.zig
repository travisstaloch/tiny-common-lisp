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
user_error: ?anyerror = null,
prims: primitives = .{},

pub fn init(cell: []Expr, w: *Io.Writer) Interp {
    var p: Interp = .{
        .memory = cell,
        .hp = 0,
        .sp = @intCast(cell.len),
        .tru = undefined,
        .env = undefined,
        .w = w,
    };
    @memset(p.memory, .{ .int = 0 });

    p.tru = p.atom("#t");
    p.env = p.pair(p.tru, p.tru, nil);
    for (0..prims_len) |i| {
        const pr: Prim = @enumFromInt(i);
        p.env = p.pair(p.atom(@tagName(pr)), box(.prim, @intCast(i)), p.env);
    }
    _ = p.run(@embedFile("common.lisp"), "common.lisp") catch |e| {
        p.abort("error.{t}.  failed to run common.lisp", .{e});
    };
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
        nil = 0x7ffc, //  0b0111111111111_100
        err = 0x7ffd, //  0b0111111111111_101

        pub fn int(t: Tag) u16 {
            return @intFromEnum(t);
        }

        pub fn isOneOf(t: Tag, comptime tags: []const u16) bool {
            const V = @Vector(tags.len, u16);
            const wanted: V = @bitCast(tags[0..tags.len].*);
            const actual: [tags.len]u16 = @splat(@intFromEnum(t));
            return @reduce(.Or, wanted == actual);
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
                PRIM => try w.print("'{t}", .{@as(Prim, @enumFromInt(ord(f.e)))}),
                CONS => {
                    var t = f.e;
                    const is_root = 1 - f.e.boxed.sign; // sign set to 1 for top level list
                    const is_quote = f.p.carAssume(t).boxed.tag == .atom and
                        std.mem.eql(u8, "quote", f.p.atomName(f.p.carAssume(t)));
                    if (is_quote) {
                        try w.writeByte('\'');
                        t = f.p.cdrAssume(t);
                        try formatList(f, w, &t);
                    } else {
                        try w.writeAll("("[0..is_root]);
                        try formatList(f, w, &t);
                        try w.writeAll(")"[0..is_root]);
                    }
                },
                CLOS => try w.print("<{d}>", .{f.e.ord()}),
                ERR_ => try w.print("error.{d}", .{f.e.ord()}),
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
const NIL_ = Expr.Tag.nil.int();
const ERR_ = Expr.Tag.err.int();

pub const nil = box(.nil, 0);

pub fn heap(i: *Interp) [*:0]u8 {
    return @ptrCast(i.memory);
}

pub fn box(t: Expr.Tag, payload: u48) Expr {
    return .{ .boxed = .{ .tag = t, .payload = payload } };
}

pub fn abort(p: *Interp, comptime fmt: []const u8, args: anytype) noreturn {
    _ = p; // autofix
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

/// parse src and then progn loop over result printing intermediates
pub fn run(p: *Interp, src: [:0]const u8, file_path: []const u8) Error!Expr {
    var t = try p.parse(src, file_path);
    var result = nil;
    while (t.boxed.tag == .cons) {
        result = try p.eval(try p.car(t), p.env);
        try p.w.print("{f}\n", .{result.fmt(p)});
        t = try p.cdr(t);
    }
    try p.w.flush();
    return result;
}

pub const Error = error{ Parse, NonPair, Unbound, CannotApply, User } ||
    Io.Reader.Error ||
    Io.Writer.Error ||
    std.fmt.ParseFloatError ||
    std.fs.File.OpenError ||
    std.fs.File.ReadError;

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
        p.err(e, "\"{f}\".\n", .{t});
}
fn parseExprInner(p: *Interp, t: *Tokenizer, token: Tokenizer.Token) Error!Expr {
    // trace("parseExpr({f})", .{token.fmt(t)});
    const src = token.src(t.src);

    return switch (token.tag) {
        .lparen => try p.parseList(t, .rparen),
        .symbol, .string => p.atom(src),
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

fn parseList(l: *Interp, t: *Tokenizer, comptime end_tag: Tokenizer.Token.Tag) Error!Expr {
    const token = t.next();
    // trace("parseList() called {f}", .{token.fmt(t)});
    switch (token.tag) {
        end_tag => return nil,
        .dot => {
            const last_expr = try l.parseExprInner(t, t.next());
            const right_paren = t.next();
            if (right_paren.tag != .rparen) return error.Parse;
            return last_expr;
        },
        else => {
            // std.debug.panic("TODO handle {t}", .{tag}),
            const first = try l.parseExprInner(t, token);
            const rest = try l.parseList(t, end_tag);
            return l.cons(first, rest);
        },
    }
}

pub fn err(p: *Interp, e: Error, comptime fmt: []const u8, args: anytype) Error {
    // TODO file:line:col
    try p.w.print("error.{t}: " ++ fmt ++ "\n", .{e} ++ args);
    return e;
}

pub fn gc(p: *Interp) void {
    p.sp = @intCast(p.env.ord());
}

fn car(p: *Interp, x: Expr) Error!Expr {
    return if (x.boxed.tag.isOneOf(&.{ CONS, CLOS }))
        p.memory[x.ord() + 1]
    else
        p.err(error.NonPair, "{f}", .{x.fmt(p)});
}

// TODO replace car with carAssume whenever CONS/CLOS is known
fn carAssume(p: *Interp, x: Expr) Expr {
    return p.car(x) catch unreachable;
}

fn carOpt(p: *Interp, x: Expr) ?Expr {
    return if (x.boxed.tag.isOneOf(&.{ CONS, CLOS }))
        p.memory[x.ord() + 1]
    else
        null;
}

fn cdr(p: *Interp, x: Expr) Error!Expr {
    return if (x.boxed.tag.isOneOf(&.{ CONS, CLOS }))
        p.memory[x.ord()]
    else
        p.err(error.NonPair, "{f}", .{x.fmt(p)});
}

fn cdrOpt(p: *Interp, x: Expr) ?Expr {
    return if (x.boxed.tag.isOneOf(&.{ CONS, CLOS }))
        p.memory[x.ord()]
    else
        null;
}

fn cdrAssume(p: *Interp, x: Expr) Expr {
    return p.cdr(x) catch unreachable;
}

/// construct a closure, returns a boxed CLOS
fn closure(p: *Interp, vars: Expr, body: Expr, env: Expr) Error!Expr {
    const pr = p.pair(vars, body, if (env.equ(p.env)) nil else env);
    return box(.clos, pr.ord());
}

/// look up a symbol in an environment, return its value or ERR if not found
fn assoc(p: *Interp, a: Expr, env: Expr) Error!Expr {
    var e = env;
    while (e.boxed.tag == .cons and !a.equ(try p.car(try p.car(e)))) {
        e = try p.cdr(e);
    }
    return if (e.boxed.tag == .cons)
        try p.cdr(try p.car(e))
    else
        p.err(error.Unbound, "{f}", .{a.fmt(p)});
}

fn let(p: *Interp, x: Expr) Error!bool {
    return !x.not() and !(try p.cdr(x)).not();
}

// create environment by extending `env` with variables `vars` bound to values `vals` */
fn bind(p: *Interp, vars: Expr, vals: Expr, env: Expr) Error!Expr {
    return switch (vars.boxed.tag.int()) {
        NIL_ => env,
        CONS => p.bind(
            try p.cdr(vars),
            try p.cdr(vals),
            p.pair(try p.car(vars), try p.car(vals), env),
        ),
        else => p.pair(vars, vals, env),
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
pub fn reduce(l: *Interp, clos: Expr, args: Expr, env: Expr) Error!Expr {
    const clos_fun = try l.car(clos);
    const clos_env = try l.cdr(clos);
    const clos_vars = try l.car(clos_fun);
    const clos_body = try l.cdr(clos_fun);
    const eval_args = try l.evlis(args, env);
    return l.eval(clos_body, try l.bind(
        clos_vars,
        eval_args,
        if (clos_env.not()) l.env else clos_env,
    ));
}

fn callPrim(pm: Prim, p: *Interp, t: Expr, env: Expr) Error!Expr {
    // trace("callPrim({t}, {f})", .{ pm, t.fmt(p) });
    return switch (pm) {
        inline else => |tag| try @field(primitives, @tagName(tag))(p, t, env),
    };
}

fn apply(p: *Interp, f: Expr, t: Expr, env: Expr) Error!Expr {
    trace("apply {f} {f}", .{ f.fmt(p), t.fmt(p) });
    return switch (f.boxed.tag.int()) {
        PRIM => try callPrim(@enumFromInt(f.ord()), p, t, env),
        CLOS => try p.reduce(f, t, env),
        else => p.err(error.CannotApply, "{f}", .{f.fmt(p)}),
    };
}

fn eval(p: *Interp, x: Expr, env: Expr) Error!Expr {
    trace("eval({f})", .{x.fmt(p)});
    return switch (x.boxed.tag.int()) {
        ATOM => try p.assoc(x, env),
        CONS => try p.apply(
            try p.eval(try p.car(x), env),
            try p.cdr(x),
            env,
        ),
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

// TODO maybe accept `p: *primitives` instead of *Interp and do
// @fieldParentPtr mixins
const primitives = struct {
    /// (eval x) return evaluated x (such as when x was quoted)
    pub fn eval(p: *Interp, t: Expr, env: Expr) Error!Expr {
        return try p.eval(try p.car(try p.evlis(t, env)), env);
    }
    /// (quote x) special form, returns x unevaluated "as is"
    pub fn quote(p: *Interp, t: Expr, _: Expr) Error!Expr {
        return p.car(t);
    }
    /// (cons x y) construct pair (x . y)
    pub fn cons(p: *Interp, t: Expr, env: Expr) Error!Expr {
        const t1 = try p.evlis(t, env);
        return p.cons(try p.car(t1), try p.car(try p.cdr(t1)));
    }
    /// (car p) car of pair p
    pub fn car(p: *Interp, t: Expr, env: Expr) Error!Expr {
        return try p.car(try p.car(try p.evlis(t, env)));
    }
    /// (cdr p) cdr of pair p
    pub fn cdr(p: *Interp, t: Expr, env: Expr) Error!Expr {
        return try p.cdr(try p.car(try p.evlis(t, env)));
    }
    const Op = enum { add, sub, mul, div };
    fn mathOp(p: *Interp, t: Expr, env: Expr, comptime op: Op) Error!Expr {
        var t1 = try p.evlis(t, env);
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
    pub fn @"+"(p: *Interp, t: Expr, env: Expr) Error!Expr {
        return mathOp(p, t, env, .add);
    }
    /// (- n1 n2 ... nk) sum of n1 to nk
    pub fn @"-"(p: *Interp, t: Expr, env: Expr) Error!Expr {
        return mathOp(p, t, env, .sub);
    }
    /// (* n1 n2 ... nk) sum of n1 to nk
    pub fn @"*"(p: *Interp, t: Expr, env: Expr) Error!Expr {
        return mathOp(p, t, env, .mul);
    }
    /// (/ n1 n2 ... nk) sum of n1 to nk
    pub fn @"/"(p: *Interp, t: Expr, env: Expr) Error!Expr {
        return mathOp(p, t, env, .div);
    }
    /// (int n) integer part of n
    pub fn int(p: *Interp, t: Expr, env: Expr) Error!Expr {
        const n = try p.car(try p.evlis(t, env));
        return .{ .float = @floatFromInt(@as(u128, @intFromFloat(n.float))) };
    }
    fn cmpOp(p: *Interp, t: Expr, env: Expr, comptime cmp: enum { lt, gt, eq }) Error!Expr {
        const t1 = try p.evlis(t, env);
        const x = try p.car(t1);
        const y = try p.car(try p.cdr(t1));
        return p.fromBool(switch (cmp) {
            .lt => x.float - y.float < 0,
            .gt => x.float - y.float > 0,
            .eq => x.equ(y),
        });
    }
    /// (< n1 n2) #t if n1<n2, otherwise ()
    pub fn @"<"(p: *Interp, t: Expr, env: Expr) Error!Expr {
        return cmpOp(p, t, env, .lt);
    }
    /// (> n1 n2) #t if n1>n2, otherwise ()
    pub fn @">"(p: *Interp, t: Expr, env: Expr) Error!Expr {
        return cmpOp(p, t, env, .gt);
    }
    /// (= n1 n2) #t if n1=n2, otherwise ()
    pub fn @"eq?"(p: *Interp, t: Expr, env: Expr) Error!Expr {
        return cmpOp(p, t, env, .eq);
    }
    pub const @"=" = @"eq?";
    /// (or x1 x2 ... xk) first x that is truthy, otherwise ()
    pub fn @"or"(p: *Interp, t: Expr, env: Expr) Error!Expr {
        var x = nil;
        var t1 = t;
        while (true) {
            if (t1.not()) break;
            x = try p.eval(try p.car(t1), env);
            if (!x.not()) break;
            t1 = try p.cdr(t1);
        }
        return x;
    }
    /// (and x1 x2 ... xk) last x if all x are truthy, otherwise ()
    pub fn @"and"(p: *Interp, t: Expr, env: Expr) Error!Expr {
        var x = nil;
        var t1 = t;
        while (true) {
            if (t1.not()) break;
            x = try p.eval(try p.car(t1), env);
            if (x.not()) break;
            t1 = try p.cdr(t1);
        }
        return x;
    }
    /// (if x y z) if x is non-() then y else z
    pub fn @"if"(p: *Interp, t: Expr, env: Expr) Error!Expr {
        const cnd = try p.eval(try p.car(t), env);
        const branch = if (cnd.not()) try p.cdr(t) else t;
        return try p.eval(try p.car(try p.cdr(branch)), env);
    }
    /// (define v x) define a named value globally
    pub fn define(p: *Interp, t: Expr, env: Expr) Error!Expr {
        const t1 = try p.cdr(t);
        const x = try p.car(t1);
        const t2 = try p.eval(x, env);
        p.env = p.pair(try p.car(t), t2, p.env);
        return try p.car(t);
    }
    pub fn cond(p: *Interp, t0: Expr, env: Expr) Error!Expr {
        var t = t0;
        while ((try p.eval(try p.car(try p.car(t)), env)).not()) {
            t = try p.cdr(t);
        }
        return try p.eval(try p.car(try p.cdr(try p.car(t))), env);
    }
    /// (not x) #t if x is (), otherwise ()
    pub fn not(p: *Interp, t: Expr, env: Expr) Error!Expr {
        return p.fromBool((try p.car(try p.evlis(t, env))).not());
    }
    /// (lambda v x) construct a closure
    pub fn lambda(p: *Interp, t: Expr, env: Expr) Error!Expr {
        return p.closure(try p.car(t), try p.car(try p.cdr(t)), env);
    }
    /// (let* ((v1 x1) (v2 x2) ... (vk xk)) y1 ... yn)
    /// eval `y`s with `v`s in env
    pub fn @"let*"(p: *Interp, t0: Expr, env: Expr) Error!Expr {
        var kvs = try p.car(t0);
        var kv = try p.car(kvs);
        var e = env;
        while (try p.let(kv)) {
            e = p.pair(
                try p.car(kv),
                try p.eval(try p.car(try p.cdr(kv)), e),
                e,
            );
            kvs = p.cdrOpt(kvs) orelse break;
            kv = p.carOpt(kvs) orelse break;
        }

        var body = p.cdrOpt(t0) orelse return nil;
        var t = p.carOpt(body) orelse return nil;
        while (true) {
            const ret = try p.eval(t, e);
            body = p.cdrOpt(body) orelse return ret;
            t = p.carOpt(body) orelse return ret;
        }
    }
    pub fn @"pair?"(p: *Interp, t: Expr, env: Expr) Error!Expr {
        const x = try p.car(try p.evlis(t, env));
        return p.fromBool(x.boxed.tag == .cons);
    }
    /// (progn (x1 x2 ...)) evaluates each x$i, returning the last one
    pub fn progn(p: *Interp, t: Expr, env: Expr) Error!Expr {
        var result = nil;
        var current = t;
        while (current.boxed.tag == .cons) {
            result = try p.eval(try p.car(current), env);
            current = try p.cdr(current);
        }
        return result;
    }
    pub fn echo(p: *Interp, t: Expr, env: Expr) Error!Expr {
        const t1 = try p.car(try p.evlis(t, env));
        try p.w.print("    >> {f}\n", .{t1.fmt(p)});
        return t1;
    }
    pub fn @"echo-eval"(p: *Interp, t: Expr, env: Expr) Error!Expr {
        const t1 = try p.car(try p.evlis(t, env));
        try p.w.print("    >> {f}\n    << {f}\n", .{ t.fmt(p), t1.fmt(p) });
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
    pub fn throw(p: *Interp, t: Expr, env: Expr) Error!Expr {
        const i = (try int(p, t, env)).float;
        try p.w.print("user error code {}\n", .{i});
        return .{ .boxed = .{ .tag = .err, .payload = @intFromFloat(i) } };
    }
    pub fn @"catch"(p: *Interp, t: Expr, env: Expr) Error!Expr {
        trace("catch {f}", .{t.fmt(p)});
        const ret = try p.eval(try p.car(t), env);
        return switch (ret.boxed.tag.int()) {
            ERR_ => {
                p.user_error = @errorFromInt(@as(u16, @intCast(ret.boxed.payload)));
                return error.User;
            },
            else => ret,
        };
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
        ERR_ => return error.ErrExpr,
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
    var l: Interp = .init(&memory, &discarding.writer);
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
    var l2: Interp = .init(&memory2, &discarding.writer);
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
    var l: Interp = .init(&memory, &discarding.writer);
    const e = try l.run(src, "<testEval>");
    try testing.expectFmt(expected, "{f}", .{e.fmt(&l)});
}

test "let*" {
    try testEval("3",
        \\(let* ((x 1) (y 2)) (let* ((z (+ x y))) z))
    );
}
test "load" {
    try testEval("begin", "(load src/common.lisp)");
}
