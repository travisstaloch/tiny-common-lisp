; fizzbuzz
(define mod (lambda (i n)
    (- i (* n (int (/ i n))))))
(mod 20 3)
(define fizzbuzz
    (lambda (i n)
        (if (not (> i n))
            (let* (x (mod i 3)) (y (mod i 5))
                (progn (echo x) (echo y) (echo i) (echo n))
                (if (eq? x 0) (echo 'Fizz))
                (if (eq? y 0) (echo 'Buzz))
                (if (and (not (eq? x 0)) (not (eq? y 0))) (echo i))
                (fizzbuzz (+ i 1) n) 
                ))
    )
)

(fizzbuzz 1 20)
