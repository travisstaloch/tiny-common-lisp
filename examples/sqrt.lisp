
(defun Y (f)
  ((lambda (g) (funcall g g))
   (lambda (g)
     (funcall f (lambda (&rest a)
		  (apply (funcall g g) a))))))

(defun fac (n)
  (funcall
   (Y (lambda (f)
       (lambda (n)
         (if (= n 0)
	         1
	         (* n (funcall f (- n 1)))))))
   n))

(print (fac 5))

; (sqrt n) -- solve x^2 - n = 0 with Newton method using the Y combinator to recurse
(defun mysqrt (n)
  (funcall
    (Y (lambda (f)
        (lambda (x)
          (let*
            ((y (- x (/ (- x (/ n x)) 2))))
            (if (= x y)
                (truncate x)
                (funcall f y))))))
  n))



(print (mysqrt 25.0))
