; fizzbuzz
(define fizzbuzz
    (lambda (i n)
        (if (not (> i n))
            (let* (x (- i (int (/ i 3)))) (y (- i (int (/ i 5))))
                ; (echo-eval x)
                ; (echo x)
                ; (echo n)
                
                (progn (echo x) (echo y))
                (if (eq? x 0) (echo 'Fizz))
                (if (eq? y 0) (echo 'Buzz))
                (if (and (not (eq? x 0)) (not (eq? y 0))) (echo i))
                (fizzbuzz (+ i 1) n) 
                ))
    )
)

(fizzbuzz 1 20)
