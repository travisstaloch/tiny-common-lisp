(load "tests/helpers.lisp")

(defun restl (a b &rest rest)
  (list a b rest))
(run-test 'rest1     '(1 2 (3 4 5)) (restl 1 2 3 4 5))

(defun opt (a &optional b)
  (list a b))
(run-test 'opt1      '(1 nil)       (opt 1 nil))
(run-test 'opt2      '(1 nil)       (opt 1))
(run-test 'opt3      '(1 2)         (opt 1 2))

(defun opt-rest (a &optional b &rest rest)
  (list a b rest))
(run-test 'opt-rest1 '(1 2 (3))     (opt-rest 1 2 3))
(run-test 'opt-rest2 '(1 2 (3 4))   (opt-rest 1 2 3 4))
