zig build test

function check() {
    echo "check $1"
    zig build run -- --banner false --script $1 > /tmp/tinycl-actual.txt
    sbcl --script $1 > /tmp/tinycl-expected.txt
    diff -u --ignore-case --ignore-all-space --ignore-space-change \
         --ignore-trailing-space --ignore-blank-lines \
         /tmp/tinycl-actual.txt /tmp/tinycl-expected.txt
}

check examples/basic.lisp
check examples/fizzbuzz.lisp
check examples/sqrt.lisp
check tests/dotcall.lisp
