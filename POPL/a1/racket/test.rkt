#lang racket

(require rackunit)
(require "main.rkt")

;; P01: my-last
(check-equal? (my-last '(a b c d)) 'd "P01-1")
(check-equal? (my-last '(1 2 3)) 3 "P01-2")
(check-equal? (my-last '(42)) 42 "P01-3")
(check-equal? (my-last '()) '() "P01-4")

;; P02: my-but-last
(check-equal? (my-but-last '(a b c d)) '(c d) "P02-1")
(check-equal? (my-but-last '(1 2)) '(1 2) "P02-2")
(check-equal? (my-but-last '(1)) '() "P02-3")
(check-equal? (my-but-last '()) '() "P02-4")

;; P03: element-at
(check-equal? (element-at '(a b c d e) 3) 'c "P03-1")
(check-equal? (element-at '(10 20 30) 1) 10 "P03-2")
(check-equal? (element-at '(10 20 30) 0) '() "P03-3")
(check-equal? (element-at '(10 20 30) 4) '() "P03-4")
(check-equal? (element-at '() 1) '() "P03-5")

;; P04: my-length
(check-equal? (my-length '(a b c)) 3 "P04-1")
(check-equal? (my-length '(1 2 3 4 5)) 5 "P04-2")
(check-equal? (my-length '()) 0 "P04-3")

;; P05: my-reverse
(check-equal? (my-reverse '(a b c)) '(c b a) "P05-1")
(check-equal? (my-reverse '(1 2 3 4)) '(4 3 2 1) "P05-2")
(check-equal? (my-reverse '()) '() "P05-3")

;; P06: palindrome?
(check-equal? (palindrome? '(x a m a x)) #t "P06-1")
(check-equal? (palindrome? '(1 2 3 2 1)) #t "P06-2")
(check-equal? (palindrome? '(1 2 3 4)) #f "P06-3")
(check-equal? (palindrome? '()) #t "P06-4")

;; P07: my-flatten
(check-equal? (my-flatten '(a (b (c d) e))) '(a b c d e) "P07-1")
(check-equal? (my-flatten '()) '() "P07-2")

;; P08: compress
(check-equal? (compress '(a a a a b c c a a d e e e e)) '(a b c a d e) "P08-1")
(check-equal? (compress '(1 1 1 2 2 3)) '(1 2 3) "P08-2")
(check-equal? (compress '(1 2 3)) '(1 2 3) "P08-3")
(check-equal? (compress '()) '() "P08-4")

;; P09: pack
(check-equal? (pack '(a a a a b c c a a d e e e e)) '((a a a a) (b) (c c) (a a) (d) (e e e e)) "P09-1")
(check-equal? (pack '(1 1 2 3 3 3)) '((1 1) (2) (3 3 3)) "P09-2")
(check-equal? (pack '()) '() "P09-3")

;; P10: encode
(check-equal? (encode '(a a a a b c c a a d e e e e)) '((4 a) (1 b) (2 c) (2 a) (1 d) (4 e)) "P10-1")
(check-equal? (encode '(1 1 2 3 3 3)) '((2 1) (1 2) (3 3)) "P10-2")
(check-equal? (encode '()) '() "P10-3")

;; P11: encode-modified
(check-equal? (encode-modified '(a a a a b c c a a d e e e e)) '((4 a) b (2 c) (2 a) d (4 e)) "P11-1")
(check-equal? (encode-modified '(1 2 2 3)) '(1 (2 2) 3) "P11-2")
(check-equal? (encode-modified '()) '() "P11-3")

;; P12: decode
(check-equal? (decode '((4 a) b (2 c) (2 a) d (4 e))) '(a a a a b c c a a d e e e e) "P12-1")
(check-equal? (decode '(1 (2 2) 3)) '(1 2 2 3) "P12-2")
(check-equal? (decode '()) '() "P12-3")

;; P13: encode-direct
(check-equal? (encode-direct '(a a a a b c c a a d e e e e)) '((4 a) b (2 c) (2 a) d (4 e)) "P13-1")
(check-equal? (encode-direct '(1 2 2 3)) '(1 (2 2) 3) "P13-2")
(check-equal? (encode-direct '()) '() "P13-3")

;; P14: dupli
(check-equal? (dupli '(a b c c d)) '(a a b b c c c c d d) "P14-1")
(check-equal? (dupli '(1 2 3)) '(1 1 2 2 3 3) "P14-2")
(check-equal? (dupli '()) '() "P14-3")

;; P15: repli
(check-equal? (repli '(a b c) 3) '(a a a b b b c c c) "P15-1")
(check-equal? (repli '(1 2) 0) '() "P15-2")
(check-equal? (repli '(1 2) 1) '(1 2) "P15-3")
(check-equal? (repli '() 3) '() "P15-4")

;; P16: drop
(check-equal? (drop '(a b c d e f g h i k) 3) '(a b d e g h k) "P16-1")
(check-equal? (drop '(1 2 3 4 5 6) 2) '(1 3 5) "P16-2")
(check-equal? (drop '(1 2 3) 0) '(1 2 3) "P16-3")
(check-equal? (drop '() 2) '() "P16-4")

;; P17: split
(check-equal? (split '(a b c d e f g h i k) 3) '((a b c) (d e f g h i k)) "P17-1")
(check-equal? (split '(1 2 3) 0) '(() (1 2 3)) "P17-2")
(check-equal? (split '(1 2 3) 5) '((1 2 3) ()) "P17-3")
(check-equal? (split '() 2) '(() ()) "P17-4")

;; P18: slice
(check-equal? (slice '(a b c d e f g h i k) 3 7) '(c d e f g) "P18-1")
(check-equal? (slice '(10 20 30 40 50) 2 4) '(20 30 40) "P18-2")
(check-equal? (slice '(10 20 30) 1 1) '(10) "P18-3")
(check-equal? (slice '() 1 3) '() "P18-4")

;; P19: rotate
(check-equal? (rotate '(a b c d e f g h) 3) '(d e f g h a b c) "P19-1")
(check-equal? (rotate '(a b c d e f g h) -2) '(g h a b c d e f) "P19-2")
(check-equal? (rotate '(1 2 3 4 5) 0) '(1 2 3 4 5) "P19-3")
(check-equal? (rotate '() 2) '() "P19-4")

;; P20: remove-at
(check-equal? (remove-at '(a b c d) 2) '(a c d) "P20-1")
(check-equal? (remove-at '(10 20 30) 1) '(20 30) "P20-2")
(check-equal? (remove-at '(10 20 30) 0) '(10 20 30) "P20-3")
(check-equal? (remove-at '(10 20 30) 5) '(10 20 30) "P20-4")
(check-equal? (remove-at '() 1) '() "P20-5")


;; P21: insert-at
(check-equal? (insert-at 'alfa '(a b c d) 2) '(a alfa b c d) "P21-1")
(check-equal? (insert-at 99 '(10 20 30) 1) '(99 10 20 30) "P21-2")
(check-equal? (insert-at 99 '(10 20 30) 4) '(10 20 30 99) "P21-3")
(check-equal? (insert-at 99 '() 1) '(99) "P21-4")

;; P22: my-range
(check-equal? (my-range 4 9) '(4 5 6 7 8 9) "P22-1")
(check-equal? (my-range 9 4) '(9 8 7 6 5 4) "P22-2")
(check-equal? (my-range 5 5) '(5) "P22-3")

;; P23: rnd-select
(check-equal? (my-length (rnd-select '(a b c d e f) 3)) 3 "P23-1")

;; P24: lotto-select
(check-equal? (my-length (lotto-select 6 49)) 6 "P24-1")

;; P25: rnd-permu
(check-equal? (my-length (rnd-permu '(a b c d))) 4 "P25-1")