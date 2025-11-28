(defun run-test (name expected actual)
  (print (cons
    (if (equal expected actual)
      'passed
      (list 'failed expected actual)) name)))
