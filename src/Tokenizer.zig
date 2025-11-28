const Tokenizer = @This();

// TODO make streaming. replace src field with buffer and reader of some kind.
// must be non seekable.
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

const std = @import("std");
const Io = std.Io;
