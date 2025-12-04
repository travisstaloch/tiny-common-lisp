(load "tests/helpers.lisp")

;; &rest
(defun restl (a b &rest rest)
  (list a b rest))
(run-test 'rest1     '(1 2 (3 4 5)) (restl 1 2 3 4 5))

;; &optional
(defun opt (a &optional b)
  (list a b))
(run-test 'opt1 '(1 nil) (opt 1 nil))
(run-test 'opt2 '(1 nil) (opt 1))
(run-test 'opt3 '(1 2)   (opt 1 2))

(defun opt-rest (a &optional b &rest rest)
  (list a b rest))
(run-test 'opt-rest1 '(1 2 (3))   (opt-rest 1 2 3))
(run-test 'opt-rest2 '(1 2 (3 4)) (opt-rest 1 2 3 4))

; TODO
;; &key - basic
; (defun keys (&key a b c)
;   (list a b c))
; (run-test 'key1 '(nil nil nil) (keys))
; (run-test 'key2 '(1 nil nil)   (keys :a 1))
; (run-test 'key3 '(1 2 nil)     (keys :a 1 :b 2))
; (run-test 'key4 '(1 2 3)       (keys :a 1 :b 2 :c 3))
; (run-test 'key5 '(1 2 3)       (keys :c 3 :b 2 :a 1)) ; order doesn't matter
; ;;      - default Values
; (defun keys-default (&key (a 10) (b 20) (c 30))
;   (list a b c))
; (run-test 'key-def1 '(10 20 30) (keys-default))
; (run-test 'key-def2 '(1 20 30)  (keys-default :a 1))
; (run-test 'key-def3 '(10 2 30)  (keys-default :b 2))
; (run-test 'key-def4 '(1 2 30)   (keys-default :a 1 :b 2))
; ;;      - mixed: required, &optional, &rest, &key
; (defun mixed (a &optional b &rest rest &key x y)
;   (list a b rest x y))
; (run-test 'mixed1 '(1 nil () nil nil)   (mixed 1))
; (run-test 'mixed2 '(1 2 () nil nil)     (mixed 1 2))
; (run-test 'mixed3 '(1 2 (3) nil nil)    (mixed 1 2 3))
; (run-test 'mixed4 '(1 2 (3) 4 nil)      (mixed 1 2 3 :x 4))
; (run-test 'mixed5 '(1 2 (3) 4 5)        (mixed 1 2 3 :x 4 :y 5))
; (run-test 'mixed6 '(1 2 (3 :x 4) 5 nil) (mixed 1 2 3 :x 4 :y 5 :x 5)) ; last x wins
; ;;      - errors - invalid key
; (run-test 'key-unknown '(1 nil) (keys :unknown 1 :a nil)) ; if unknown ignored
