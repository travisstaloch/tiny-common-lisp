(defun run-test (name expected actual)
  (print (cons
    (if (equal expected actual)
      'passed
      'failed) name)))

(run-test 'apply
  6
  (apply '+ '(1 2 3)))

(run-test '+
  6
  ((lambda (l) (apply '+ l)) '(1 2 3)))

(run-test '-
  -4
  ((lambda (l) (apply '- l)) '(1 2 3)))

(run-test '*
  6
  ((lambda (l) (apply '* l)) '(1 2 3)))

(run-test '/
  (/ 1 2)
  ((lambda (l) (apply '/ l)) '(1 2)))

(run-test 'let*
   3
   (let*
       ((x 1) (y (+ 1 x)))
       (let* ((z (+ x y))) z)))

(run-test 'static-scoping
  3
  (let*
      ((x 2))
      ((lambda (y)
         (let* ((f '+) (x 1))
           (apply f (list x y))))
       x)))

(run-test 'currying
  15
  (let*
    ((adder (lambda (n) (lambda (x) (+ x n))))
     (add5 (apply adder '(5))))
    (apply add5 '(10))))

(run-test 'same-args
  '((1) (2) (3))
  ((lambda (x y z) (list x y z)) '(1) '(2) '(3)))

; y combinator - https://rosettacode.org/wiki/Y_combinator#Common_Lisp
(defun Y (f)
  ((lambda (g) (funcall g g))
   (lambda (g)
     (funcall f (lambda (&rest a)
		  (apply (funcall g g) a))))))

(defun y-factorial (n)
  (funcall
   (Y (lambda (f)
       (lambda (n)
         (if (= 0 n)
	         1
	         (* n (funcall f (- n 1)))))))
   n))

(run-test 'y-factorial 120 (y-factorial 5))

; (sqrt n) -- solve x^2 - n = 0 with Newton method
(defun y-sqrt (n)
  (funcall
    (Y (lambda (f)
        (lambda (x)
          (let*
            ((y (- x (/ (- x (/ n x)) 2))))
            (if (= x y)
                x
                (funcall f y))))))
  n))

(run-test 'y-sqrt 5.0 (y-sqrt 25.0))

(defun rest-as-list (a b &rest rest) (list a b rest))
(run-test 'rest-params '(1 2 (3 4 5)) (rest-as-list 1 2 3 4 5))

(print 'OK)
