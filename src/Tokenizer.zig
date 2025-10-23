const std = @import("std");
const assert = std.debug.assert;

src: [:0]const u8,
pos: u32 = 0,
mode: struct {
    whitespace: Emit = .skip,
    comments: Emit = .skip,
    const Emit = enum { emit, skip };
} = .{},

const Tokenizer = @This();

pub const Token = struct {
    start: u32,
    end: u32,
    tag: Tag,

    pub const empty: Token = .{ .start = 0, .end = 0, .tag = .invalid };

    pub const Tag = enum {
        invalid,
        eof,
        comment,
        whitespace,
        lparen,
        rparen,
        symbol,
        /// A double quoted string. The quotes are included
        /// in the string.
        string,
        /// A number literal.
        number,
        /// '
        quote,
        /// `
        quasi_quote,
        /// ,
        quasi_unquote,
        /// ,@
        quasi_unquote_splicing,
    };

    pub fn src(self: Token, s: [:0]const u8) []const u8 {
        return s[self.start..self.end];
    }

    pub fn line(self: Token, s: [:0]const u8) []const u8 {
        var start = self.start;
        while (start != 0) {
            start -= 1;
            if (s[start] == '\n') break;
        }
        var end = self.end;
        while (end < s.len) : (end += 1) {
            if (s[end] == '\n') break;
        }
        return s[start + @intFromBool(start != 0) .. end];
    }

    pub const Fmt = struct {
        src: [:0]const u8,
        token: Token,

        pub fn format(
            self: @This(),
            writer: *std.Io.Writer,
        ) std.Io.Writer.Error!void {
            const s = self.token.src(self.src);
            if (self.token.tag == .string) {
                try std.zig.stringEscape(s, writer);
            } else try writer.writeAll(s);
        }
    };

    pub fn fmt(self: Token, s: [:0]const u8) Fmt {
        return .{ .token = self, .src = s };
    }
};

fn isIdent(byte: u8) bool {
    return byte != ')' and byte != '(' and
        !std.ascii.isDigit(byte) and
        !isWhitespace(byte) and
        std.ascii.isPrint(byte);
}

fn isIdent2(byte: u8) bool {
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

fn peek(self: *Tokenizer, offset: usize) u8 {
    return self.src[self.pos + offset];
}

fn advance(self: *Tokenizer, n: u32) void {
    self.pos += n;
}

fn advanceTo(self: *Tokenizer, tag: @Type(.enum_literal), n: u32, t: *Token) State {
    self.advance(n);
    if (@hasField(Token.Tag, @tagName(tag)) and tag != .number) t.tag = tag;
    return switch (tag) {
        .lparen,
        .rparen,
        .invalid,
        .quote,
        .quasi_quote,
        .quasi_unquote,
        .quasi_unquote_splicing,
        => .init,
        else => |ttag| @field(State, @tagName(ttag)),
    };
}

const State = enum {
    init,
    comment,
    symbol,
    whitespace,
    number,
    string,
    string_escape,
};

pub fn next(self: *Tokenizer) Token {
    // std.debug.print("next> pos {}\n", .{self.pos});
    // defer std.debug.print("next< pos {}\n", .{self.pos});
    var t: Token = .{ .start = self.pos, .end = self.pos, .tag = .invalid };

    state: switch (State.init) {
        inline else => |state| {
            const byte = self.peek(0);
            // std.debug.print("  state {t} byte '{c}'\n", .{ state, byte });
            switch (state) {
                .init => switch (byte) {
                    0 => {
                        t.tag = .eof;
                        break :state;
                    },
                    ';' => continue :state self.advanceTo(.comment, 1, &t),
                    '(' => _ = self.advanceTo(.lparen, 1, &t),
                    ')' => _ = self.advanceTo(.rparen, 1, &t),
                    '\'' => _ = self.advanceTo(.quote, 1, &t),
                    '`' => _ = self.advanceTo(.quasi_quote, 1, &t),
                    ',' => _ = if (self.peek(1) == '@')
                        self.advanceTo(.quasi_unquote_splicing, 2, &t)
                    else
                        self.advanceTo(.quasi_unquote, 1, &t),
                    '0'...'9' => continue :state self.advanceTo(.number, 1, &t),
                    '-', '+' => {
                        switch (self.peek(1)) {
                            '1'...'9' => continue :state self.advanceTo(.number, 1, &t),
                            '0' => self.pos += 2,
                            else => continue :state self.advanceTo(.symbol, 1, &t),
                        }
                    },
                    '"' => continue :state self.advanceTo(.string, 1, &t),
                    else => if (isIdent(byte)) {
                        continue :state self.advanceTo(.symbol, 1, &t);
                    } else if (isWhitespace(byte)) {
                        continue :state self.advanceTo(.whitespace, 1, &t);
                    } else std.debug.panic("unexpected byte '{c}'\n", .{byte}),
                },
                .whitespace => if (isWhitespace(byte)) {
                    continue :state self.advanceTo(.whitespace, 1, &t);
                } else {
                    if (self.mode.whitespace == .skip) {
                        t = .{
                            .tag = .invalid,
                            .start = self.pos,
                            .end = self.pos,
                        };
                        continue :state .init;
                    }
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
                        // else => std.debug.panic("unexpected byte '{c}'\n", .{byte}),
                    }
                    if (self.mode.comments == .skip) {
                        t = .{
                            .tag = .invalid,
                            .start = self.pos,
                            .end = self.pos,
                        };
                        continue :state .init;
                    }
                },
                .symbol => if (isIdent2(byte)) {
                    continue :state self.advanceTo(.symbol, 1, &t);
                },
                .number => if (std.ascii.isDigit(byte)) {
                    continue :state self.advanceTo(.number, 1, &t);
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
    return t;
}

const testing = std.testing;

test "tokenize an expression" {
    const src =
        \\ (begin (print "str" 5 `(,a ,@b)))
    ;
    var t: Tokenizer = .{ .src = src };
    try testing.expectEqual(.lparen, t.next().tag);
    {
        const tok = t.next();
        try testing.expectEqual(.symbol, tok.tag);
        try testing.expectEqualStrings("begin", tok.src(src));
    }
    try testing.expectEqual(.lparen, t.next().tag);
    try testing.expectEqual(.symbol, t.next().tag);
    try testing.expectEqual(.string, t.next().tag);
    {
        const tok = t.next();
        try testing.expectEqual(.number, tok.tag);
        try testing.expectEqualStrings("5", tok.src(src));
    }
    try testing.expectEqual(.quasi_quote, t.next().tag);
    try testing.expectEqual(.lparen, t.next().tag);
    try testing.expectEqual(.quasi_unquote, t.next().tag);
    try testing.expectEqual(.symbol, t.next().tag);
    try testing.expectEqual(.quasi_unquote_splicing, t.next().tag);
    try testing.expectEqual(.symbol, t.next().tag);
    try testing.expectEqual(.rparen, t.next().tag);
    try testing.expectEqual(.rparen, t.next().tag);
    try testing.expectEqual(.rparen, t.next().tag);
    try testing.expectEqual(.eof, t.next().tag);
    try testing.expectEqual(.eof, t.next().tag);
}
