; fizzbuzz
(define mod (lambda (i n)
  (- i (* n (int (/ i n))))))

(cons (if (and
  (= 1 (mod 19 3))
  (= 2 (mod 20 3))
  (= 0 (mod 21 3))
  ) 'passed 'failed) '(mod))

(define fizzbuzz
  (lambda (i n)
    (if (<= i n)
      (let* ((x (mod i 3)) (y (mod i 5)))
        ; (echo x y i n)
        (if (or (= x 0) (= y 0))
          (if (and (= x 0) (= y 0))
            (echo 'FizzBuzz)
            (if (= x 0)
              (echo 'Fizz)
              (echo 'Buzz)))
          (echo i))
        (fizzbuzz (+ i 1) n))
    #f)
  )
)

(fizzbuzz 1 20)
