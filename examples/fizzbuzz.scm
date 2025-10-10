; fizzbuzz
(define (fizzbuzz i n)
    (if (<= i n)
        (let ((x (@mod i 3)) (y (@mod i 5)))
            (if (= x 0) (println "Fizz"))
            (if (= y 0) (println "Buzz"))
            (if (and (!= x 0) (!= y 0)) (println i))
            (fizzbuzz (+ i 1) n)
        )
    )
)

(fizzbuzz (1 20))
