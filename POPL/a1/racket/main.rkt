#lang racket

(require racket/random)

;; =============================================================================
;; CS1.402 Principles of Programming Languages (PoPL 2026-Monsoon @ IIITH)
;; Assignment 1: Functional Programming & List Processing (Racket Component)
;; =============================================================================
;;
;; Instructions:
;; Implement functions P01 through P25 below.
;;
;; Restrictions:
;; Do NOT use forbidden built-in functions such as `length`, `reverse`, `drop`,
;; `take`, `flatten`, `list-ref`, `set!`, or `list-set`. Write recursive
;; procedures using basic primitives (`car`, `cdr`, `cons`, `cond`, `match`).
;; =============================================================================

;; P01 (*): Find the last element of a list.
(define (my-last l)
  (cond
    [(null? l) '()]
    [(null? (cdr l)) (car l)]
    [else (my-last (cdr l))]))

;; P02 (*): Find the last but one element of a list.
(define (my-but-last l)
  (cond
    [(null? l) '()]
    [(null? (cdr l)) l]
    [(null? (cddr l)) l]
    [else (my-but-last (cdr l))]))

;; P03 (*): Find the K'th element of a list (1-indexed).
(define (element-at l k)
  (cond
    [(null? l) '()]
    [(<= k 0) '()]
    [(= k 1) (car l)]
    [else (element-at (cdr l) (- k 1))]))

;; P04 (*): Find the number of elements of a list.
(define (my-length l)
  (cond
    [(null? l) 0]
    [else (+ 1 (my-length (cdr l)))]))

;; P05 (*): Reverse a list.
(define (my-reverse l)
  (cond
    [(null? l) '()]
    [else (append (my-reverse (cdr l)) (list (car l)))]))

;; P06 (*): Find out whether a list is a palindrome.
(define (palindrome? l)
  (equal? l (my-reverse l)))

;; P07 (**): Flatten a nested list structure.
(define (my-flatten lst)
  (cond
    [(null? lst) '()]
    [(pair? (car lst)) (append (my-flatten (car lst)) (my-flatten (cdr lst)))]
    [else (cons (car lst) (my-flatten (cdr lst)))]))

;; P08 (**): Eliminate consecutive duplicates of list elements.
(define (compress lst)
  (cond
    [(null? lst) '()]
    [(null? (cdr lst)) lst]
    [(equal? (car lst) (cadr lst)) (compress (cdr lst))]
    [else (cons (car lst) (compress (cdr lst)))]))

;; P09 (**): Pack consecutive duplicates of list elements into sublists.
(define (pack lst)
  (cond
    [(null? lst) '()]
    [else
     (let ([rest (pack (cdr lst))])
       (cond
         [(null? rest) (list (list (car lst)))]
         [(equal? (car lst) (caar rest))
          (cons (cons (car lst) (car rest)) (cdr rest))]
         [else
          (cons (list (car lst)) rest)]))]))

;; P10 (**): Run-length encoding of a list.
(define (encode lst)
  (cond
    [(null? lst) '()]
    [else
     (let ([rest (encode (cdr lst))])
       (cond
         [(null? rest) (list (list 1 (car lst)))]
         [(equal? (car lst) (cadar rest))
          (cons (list (+ 1 (caar rest)) (car lst)) (cdr rest))]
         [else
          (cons (list 1 (car lst)) rest)]))]))

;; P11 (**): Modified run-length encoding.
(define (encode-modified lst)
  (let loop ([enc (encode lst)])
    (cond
      [(null? enc) '()]
      [(= (caar enc) 1) (cons (cadar enc) (loop (cdr enc)))]
      [else (cons (car enc) (loop (cdr enc)))])))

;; P12 (**): Decode a run-length encoded list.
(define (decode lst)
  (letrec ([repeat (lambda (n x)
                     (if (<= n 0) '() (cons x (repeat (- n 1) x))))])
    (cond
      [(null? lst) '()]
      [(pair? (car lst))
       (append (repeat (caar lst) (cadar lst)) (decode (cdr lst)))]
      [else
       (cons (car lst) (decode (cdr lst)))])))

;; P13 (**): Run-length encoding of a list (direct solution).
(define (encode-direct lst)
  (cond
    [(null? lst) '()]
    [else
     (let ([rest (encode-direct (cdr lst))])
       (cond
         [(null? rest) (list (car lst))]
         [(pair? (car rest))
          (if (equal? (car lst) (cadar rest))
              (cons (list (+ 1 (caar rest)) (car lst)) (cdr rest))
              (cons (car lst) rest))]
         [else
          (if (equal? (car lst) (car rest))
              (cons (list 2 (car lst)) (cdr rest))
              (cons (car lst) rest))]))]))

;; P14 (*): Duplicate the elements of a list.
(define (dupli lst)
  (cond
    [(null? lst) '()]
    [else (cons (car lst) (cons (car lst) (dupli (cdr lst))))]))

;; P15 (**): Replicate the elements of a list a given number of times.
(define (repli lst n)
  (letrec ([repeat (lambda (k x)
                     (if (<= k 0) '() (cons x (repeat (- k 1) x))))])
    (cond
      [(null? lst) '()]
      [else (append (repeat n (car lst)) (repli (cdr lst) n))])))

;; P16 (**): Drop every N'th element from a list (1-indexed).
(define (drop lst n)
  (if (<= n 0) lst
      (let loop ([l lst] [count 1])
        (cond
          [(null? l) '()]
          [(= count n) (loop (cdr l) 1)]
          [else (cons (car l) (loop (cdr l) (+ count 1)))]))))

;; P17 (*): Split a list into two parts; the length of the first part is given.
(define (split lst n)
  (cond
    [(null? lst) '(() ())]
    [(<= n 0) (list '() lst)]
    [else
     (let ([res (split (cdr lst) (- n 1))])
       (list (cons (car lst) (car res)) (cadr res)))]))

;; P18 (**): Extract a slice from a list (1-indexed, s and e inclusive).
(define (slice lst s e)
  (let loop ([l lst] [i 1])
    (cond
      [(null? l) '()]
      [(> i e) '()]
      [(>= i s) (cons (car l) (loop (cdr l) (+ i 1)))]
      [else (loop (cdr l) (+ i 1))])))

;; P19 (**): Rotate a list N places to the left.
(define (rotate lst n)
  (let ([len (my-length lst)])
    (if (= len 0) '()
        (let* ([k (modulo (+ (modulo n len) len) len)]
               [sp (split lst k)])
          (append (cadr sp) (car sp))))))

;; P20 (*): Remove the K'th element from a list (1-indexed).
(define (remove-at lst n)
  (cond
    [(null? lst) '()]
    [(<= n 0) lst]
    [(= n 1) (cdr lst)]
    [else (cons (car lst) (remove-at (cdr lst) (- n 1)))]))

;; P21 (*): Insert an element at a given position into a list (1-indexed).
(define (insert-at e lst i)
  (cond
    [(null? lst) (list e)]
    [(<= i 1) (cons e lst)]
    [else (cons (car lst) (insert-at e (cdr lst) (- i 1)))]))

;; P22 (*): Create a list containing all integers within a given range.
(define (my-range s e)
  (cond
    [(= s e) (list s)]
    [(< s e) (cons s (my-range (+ s 1) e))]
    [else (cons s (my-range (- s 1) e))]))

;; P23 (**): Extract a given number of randomly selected elements from a list.
(define (rnd-select lst x)
  (cond
    [(or (<= x 0) (null? lst)) '()]
    [else
     (let* ([len (my-length lst)]
            [idx (+ 1 (random len))]
            [elem (element-at lst idx)]
            [rest-lst (remove-at lst idx)])
       (cons elem (rnd-select rest-lst (- x 1))))]))

;; P24 (*): Lotto: Draw N different random numbers from the set 1..M.
(define (lotto-select x y)
  (rnd-select (my-range 1 y) x))

;; P25 (*): Generate a random permutation of the elements of a list.
(define (rnd-permu lst)
  (rnd-select lst (my-length lst)))

(provide (all-defined-out))
