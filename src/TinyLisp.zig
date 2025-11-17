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
err: Expr,
env: Expr,
w: *Io.Writer,

pub fn init(cell: []Expr, w: *Io.Writer) Interp {
    var p: Interp = .{
        .memory = cell,
        .hp = 0,
        .sp = @intCast(cell.len),
        .tru = undefined,
        .err = undefined,
        .env = undefined,
        .w = w,
    };
    @memset(p.memory, .{ .int = 0 });

    p.err = p.atom("ERR");
    p.tru = p.atom("#t");
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

    pub const Tag = enum(u15) {
        atom = 0x7ff8, // 0b0111111111111_000
        prim = 0x7ff9, // 0b0111111111111_001
        cons = 0x7ffa, // 0b0111111111111_010
        clos = 0x7ffb, // 0b0111111111111_011
        nil = 0x7ffc, //  0b0111111111111_100

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
        return x.boxed.tag == .nil;
    }

    pub fn equ(x: Expr, y: Expr) bool {
        return x.int == y.int;
    }

    pub const Fmt = struct {
        p: *Interp,
        e: Expr,

        pub fn format(f: Fmt, w: *Io.Writer) !void {
            switch (f.e.boxed.tag.int()) {
                NIL_ => try w.writeAll("()"),
                ATOM => try w.writeAll(f.p.atomName(f.e)),
                PRIM => try w.print("'{t}", .{@as(Prim, @enumFromInt(ord(f.e)))}),
                CONS => {
                    var t = f.e;
                    const is_root = 1 - f.e.boxed.sign; // sign set to 1 for top level list
                    try w.writeAll("("[0..is_root]);
                    while (true) {
                        try w.print("{f}", .{f.p.car(t).fmt(f.p)});
                        t = f.p.cdr(t);
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
                    try w.writeAll(")"[0..is_root]);
                },
                CLOS => try w.print("<{d}>", .{f.e.ord()}),
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
pub fn run(p: *Interp, src: [:0]const u8, file_path: []const u8) !Expr {
    var t = try p.parse(src, file_path);
    var result = nil;
    while (t.boxed.tag == .cons) {
        result = p.eval(p.car(t), p.env);
        try p.w.print("{: >4}> {f}\n", .{ p.sp - p.hp / 8, result.fmt(p) });
        t = p.cdr(t);
    }
    try p.w.flush();
    return result;
}

pub const Error = error{Parse} ||
    Io.Reader.Error ||
    Io.Writer.Error ||
    std.fmt.ParseFloatError;

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
    return p.parseExprInner(t, token) catch |e| {
        switch (e) {
            error.Parse => try p.w.print("[ERROR] Failed to parse \"{f}\".\n", .{t}),
            else => {},
        }
        return e;
    };
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

fn dumpErr(p: *Interp) Expr {
    // p.printEnv(p.env) catch {};
    // p.printStack() catch {};
    // p.printHeap() catch {};
    return p.err;
}

pub fn gc(p: *Interp) void {
    p.sp = @intCast(p.env.ord());
}

fn car(p: *Interp, x: Expr) Expr {
    return if (x.boxed.tag.isOneOf(&.{ CONS, CLOS }))
        p.memory[x.ord() + 1]
    else
        p.dumpErr();
}

fn cdr(p: *Interp, x: Expr) Expr {
    return if (x.boxed.tag.isOneOf(&.{ CONS, CLOS }))
        p.memory[x.ord()]
    else
        p.dumpErr();
}

/// construct a closure, returns a boxed CLOS
fn closure(p: *Interp, vars: Expr, body: Expr, env: Expr) Expr {
    const pr = p.pair(vars, body, if (env.equ(p.env)) nil else env);
    return box(.clos, pr.ord());
}

/// look up a symbol in an environment, return its value or ERR if not found
fn assoc(p: *Interp, a: Expr, env: Expr) Expr {
    var e = env;
    while (e.boxed.tag == .cons and !a.equ(p.car(p.car(e)))) {
        e = p.cdr(e);
    }
    return if (e.boxed.tag == .cons) p.cdr(p.car(e)) else p.dumpErr();
}

fn let(p: *Interp, x: Expr) bool {
    return !x.not() and !p.cdr(x).not();
}

// create environment by extending `env` with variables `vars` bound to values `vals` */
fn bind(p: *Interp, vars: Expr, vals: Expr, env: Expr) Expr {
    return switch (vars.boxed.tag.int()) {
        NIL_ => env,
        CONS => p.bind(
            p.cdr(vars),
            p.cdr(vals),
            p.pair(p.car(vars), p.car(vals), env),
        ),
        else => p.pair(vars, vals, env),
    };
}

fn evlis(p: *Interp, t: Expr, e: Expr) Expr {
    return switch (t.boxed.tag.int()) {
        CONS => p.cons(p.eval(p.car(t), e), p.evlis(p.cdr(t), e)),
        ATOM => p.assoc(t, e),
        else => nil,
    };
}

/// apply closure `clos` to arguments `args` in environment `env`
pub fn reduce(l: *Interp, clos: Expr, args: Expr, env: Expr) Expr {
    const clos_fun = l.car(clos);
    const clos_env = l.cdr(clos);
    const clos_vars = l.car(clos_fun);
    const clos_body = l.cdr(clos_fun);
    const eval_args = l.evlis(args, env);
    return l.eval(clos_body, l.bind(
        clos_vars,
        eval_args,
        if (clos_env.not()) l.env else clos_env,
    ));
}

fn callPrim(pm: Prim, p: *Interp, t: Expr, env: Expr) Error!Expr {
    // trace("callPrim({t}, {f})", .{ pm, t.fmt(p) });
    return switch (pm) {
        inline else => |tag| @field(primitives, @tagName(tag))(p, t, env),
    };
}

fn apply(p: *Interp, f: Expr, t: Expr, env: Expr) Expr {
    trace("apply({f}, {f})", .{ f.fmt(p), t.fmt(p) });
    return switch (f.boxed.tag.int()) {
        PRIM => blk: {
            const pm: Prim = @enumFromInt(f.ord());
            break :blk callPrim(pm, p, t, env) catch
                p.abort("error in primitive '{t}'", .{pm});
        },
        CLOS => p.reduce(f, t, env),
        else => p.dumpErr(),
    };
}

fn eval(p: *Interp, x: Expr, env: Expr) Expr {
    trace("eval({f})", .{x.fmt(p)});
    return switch (x.boxed.tag.int()) {
        ATOM => p.assoc(x, env),
        CONS => blk: {
            const xx = p.eval(p.car(x), env);
            break :blk p.apply(xx, p.cdr(x), env);
        },
        else => x,
    };
}

fn printHeap(l: *Interp) !void {
    const max_symbol_len = 20;

    try l.w.writeAll(
        \\------------------- HEAP -------------------
        \\|  #  |  address |  symbol                 |
        \\|-----|----------|-------------------------|
        \\
    );

    var atom_count: usize = 0;
    var last_i: usize = 0;

    var trimmed_i: usize = undefined;
    var symbol_suffix: *const [3:0]u8 = undefined;
    for (l.heap()[0..l.hp], 0..) |byte, i| {
        if (byte != 0) continue;

        if (i - last_i <= max_symbol_len) {
            trimmed_i = i;
            symbol_suffix = "   ";
        } else {
            trimmed_i = last_i + max_symbol_len;
            symbol_suffix = "...";
        }
        try l.w.print("| {:>3} |  0x{X:0>4}  |  {s:<20}{s}|\n", .{
            atom_count,
            last_i,
            l.heap()[last_i..trimmed_i :0],
            symbol_suffix,
        });

        atom_count += 1;
        last_i = i + 1;
    }
    try l.w.writeAll(
        \\|                    ...                   |
        \\--------------------------------------------
        \\
    );
}

fn printStack(p: *Interp) !void {
    try p.w.writeAll(
        \\------------- STACK ------------
        \\|  addr |   tag  |  ordinal |     Expr     
        \\|-------|--------|----------|--------------
        \\
    );

    var sp = p.memory.len - (prims_len + 1) * 4;
    while (sp > p.sp) {
        try p.w.print("| {:>5} |", .{sp});
        sp -= 1;
        const x = p.memory[sp];
        switch (x.boxed.tag.int()) {
            NIL_ => try p.w.print("  NIL   |   {:>5}  |  {s}\n", .{ x.ord(), "()" }),
            ATOM => try p.w.print("  ATOM  |  0x{X:0>4}  |  {s}\n", .{ x.ord(), p.atomName(x) }),
            PRIM => try p.w.print("  PRIM  |   {:>5}  |  '{t}\n", .{ x.ord(), @as(Prim, @enumFromInt(x.ord())) }),
            CONS => try p.w.print("  CONS  |   {:>5}  |\n", .{x.ord()}),
            CLOS => try p.w.print("  CLOS  |   {:>5}  |\n", .{x.ord()}),
            else => try p.w.print("        |          |  {d}\n", .{x.float}),
        }
    }
    try p.w.writeAll(
        \\|             ...              |
        \\|------------------------------|
        \\
    );
}

fn printEnv(p: *Interp, env: Expr) !void {
    var e = env;
    try p.w.writeAll(
        \\
        \\ENV
        \\(
        \\
    );
    while (!e.not()) {
        const pp = p.car(e);
        try p.w.print("\t{f}\n", .{pp.fmt(p)});
        e = p.cdr(e);
    }
    try p.w.writeAll(")\n");
}

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

const primitives = struct {
    /// (eval x) return evaluated x (such as when x was quoted)
    pub fn eval(p: *Interp, t: Expr, env: Expr) Error!Expr {
        return p.eval(p.car(p.evlis(t, env)), env);
    }
    /// (quote x) special form, returns x unevaluated "as is"
    pub fn quote(p: *Interp, t: Expr, _: Expr) Error!Expr {
        return p.car(t);
    }
    /// (cons x y) construct pair (x . y)
    pub fn cons(p: *Interp, t: Expr, env: Expr) Error!Expr {
        const t1 = p.evlis(t, env);
        return p.cons(p.car(t1), p.car(p.cdr(t1)));
    }
    /// (car p) car of pair p
    pub fn car(p: *Interp, t: Expr, env: Expr) Error!Expr {
        return p.car(p.car(p.evlis(t, env)));
    }
    /// (cdr p) cdr of pair p
    pub fn cdr(p: *Interp, t: Expr, env: Expr) Error!Expr {
        return p.cdr(p.car(p.evlis(t, env)));
    }
    const Op = enum { add, sub, mul, div };
    fn mathOp(p: *Interp, t: Expr, env: Expr, comptime op: Op) Error!Expr {
        var t1 = p.evlis(t, env);
        var n = p.car(t1);
        while (true) {
            t1 = p.cdr(t1);
            if (t1.not()) break;
            switch (op) {
                .add => n.float += p.car(t1).toNum().float,
                .sub => n.float -= p.car(t1).toNum().float,
                .mul => n.float *= p.car(t1).toNum().float,
                .div => n.float /= p.car(t1).toNum().float,
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
        const n = p.car(p.evlis(t, env));
        // TODO return n<1e16 && n>-1e16 ? (long long)n : n;

        return .{ .float = @floatFromInt(@as(u128, @intFromFloat(n.float))) };
    }
    fn cmpOp(p: *Interp, t: Expr, env: Expr, comptime cmp: enum { lt, gt, eq }) Error!Expr {
        const t1 = p.evlis(t, env);
        const x = p.car(t1);
        const y = p.car(p.cdr(t1));
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
    /// (or x1 x2 ... xk) first x that is truthy, otherwise ()
    pub fn @"or"(p: *Interp, t: Expr, env: Expr) Error!Expr {
        var x = nil;
        var t1 = t;
        while (true) {
            if (t1.not()) break;
            x = p.eval(p.car(t1), env);
            if (!x.not()) break;
            t1 = p.cdr(t1);
        }
        return x;
    }
    /// (and x1 x2 ... xk) last x if all x are truthy, otherwise ()
    pub fn @"and"(p: *Interp, t: Expr, env: Expr) Error!Expr {
        var x = nil;
        var t1 = t;
        while (true) {
            if (t1.not()) break;
            x = p.eval(p.car(t1), env);
            if (x.not()) break;
            t1 = p.cdr(t1);
        }
        return x;
    }
    /// (if x y z) if x is non-() then y else z
    pub fn @"if"(p: *Interp, t: Expr, env: Expr) Error!Expr {
        const cnd = p.eval(p.car(t), env);
        const branch = if (cnd.not()) p.cdr(t) else t;
        return p.eval(p.car(p.cdr(branch)), env);
    }
    /// (define v x) define a named value globally
    pub fn define(p: *Interp, t: Expr, env: Expr) Error!Expr {
        const t1 = p.cdr(t);
        const x = p.car(t1);
        const t2 = p.eval(x, env);
        p.env = p.pair(p.car(t), t2, p.env);
        return p.car(t);
    }
    pub fn cond(p: *Interp, t0: Expr, env: Expr) Error!Expr {
        var t = t0;
        while (p.eval(p.car(p.car(t)), env).not()) {
            t = p.cdr(t);
        }
        return p.eval(p.car(p.cdr(p.car(t))), env);
    }
    /// (not x) #t if x is (), otherwise ()
    pub fn not(p: *Interp, t: Expr, env: Expr) Error!Expr {
        return p.fromBool(p.car(p.evlis(t, env)).not());
    }
    /// (lambda v x) construct a closure
    pub fn lambda(p: *Interp, t: Expr, env: Expr) Error!Expr {
        return p.closure(p.car(t), p.car(p.cdr(t)), env);
    }
    pub fn @"let*"(p: *Interp, t0: Expr, env: Expr) Error!Expr {
        var t = t0;
        var e = env;
        while (p.let(t)) : (t = p.cdr(t)) {
            e = p.pair(
                p.car(p.car(t)),
                p.eval(p.car(p.cdr(p.car(t))), e),
                e,
            );
        }
        return p.eval(p.car(t), e);
    }
    pub fn @"pair?"(p: *Interp, t: Expr, env: Expr) Error!Expr {
        const x = p.car(p.evlis(t, env));
        return p.fromBool(x.boxed.tag == .cons);
    }
    /// (progn (x1 x2 ...)) evaluates each x$i, returning the last one
    pub fn progn(p: *Interp, t: Expr, env: Expr) Expr {
        var result = nil;
        var current = t;
        while (current.boxed.tag == .cons) {
            result = p.eval(p.car(current), env);
            current = p.cdr(current);
        }
        return result;
    }

    pub fn @"print-heap"(p: *Interp, _: Expr, _: Expr) Error!Expr {
        try p.printHeap();
        return nil;
    }
    pub fn @"print-stack"(p: *Interp, _: Expr, _: Expr) Error!Expr {
        try p.printStack();
        return nil;
    }
    pub fn @"print-env"(p: *Interp, _: Expr, env: Expr) Error!Expr {
        try p.printEnv(env);
        return nil;
    }
    pub fn echo(p: *Interp, t: Expr, env: Expr) Error!Expr {
        const t1 = p.car(p.evlis(t, env));
        try p.w.print("    >> {f}\n", .{t1.fmt(p)});
        return t1;
    }
    pub fn @"echo-eval"(p: *Interp, t: Expr, env: Expr) Error!Expr {
        const t1 = p.car(p.evlis(t, env));
        try p.w.print("    >> {f}\n    << {f}\n", .{ t.fmt(p), t1.fmt(p) });
        return t1;
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
        ATOM => {
            if (expected.e.equ(expected.p.err)) return error.ErrExpr;
            try testing.expectEqualStrings(
                ep.atomName(expected.e),
                ap.atomName(actual.e),
            );
        },
        CONS => {
            try expectExprEqual(
                ep.car(expected.e).fmt(ep),
                ap.car(actual.e).fmt(ap),
            );
            try expectExprEqual(
                ep.cdr(expected.e).fmt(ep),
                ap.cdr(actual.e).fmt(ap),
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
