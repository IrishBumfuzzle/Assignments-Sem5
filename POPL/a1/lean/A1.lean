-- =============================================================================
-- CS1.402 Principles of Programming Languages (PoPL 2026-Monsoon @ IIITH)
-- Assignment 1: Functional Programming & List Processing (Lean 4 Component)
-- =============================================================================

namespace A1

-- Algebraic Data Type for P07 (Nested List Structure)
inductive NestedList (α : Type) where
  | elem : α → NestedList α
  | list : List (NestedList α) → NestedList α

-- Algebraic Data Type for P11, P12, P13 (Modified Run-Length Encoding Item)
inductive RLEItem (α : Type) where
  | single : α → RLEItem α
  | run : Nat → α → RLEItem α
  deriving BEq, Repr

-- P01 (*): Find the last element of a list.
def myLast {α : Type} (l : List α) : Option α :=
  match l with
  | [] => none
  | [x] => x
  | _ :: xs => myLast xs

-- P02 (*): Find the last but one element of a list.
def myButLast {α : Type} (l : List α) : Option (List α) :=
  match l with
  | [] => none
  | [x] => [x]
  | [x,y] => [x, y]
  | _ :: xs => myButLast xs

-- P03 (*): Find the K'th element of a list (1-indexed).
def elementAt {α : Type} (l : List α) (k : Nat) : Option α :=
  match l, k with
  | [], _ => none
  | _, 0 => none
  | x :: _, 1 => x
  | _ :: xs, k' + 1 => elementAt xs k'

-- P04 (*): Find the number of elements of a list.
def myLength {α : Type} (l : List α) : Nat :=
  match l with
  | [] => 0
  | _ :: x => myLength x + 1

-- P05 (*): Reverse a list.
def myReverse {α : Type} (l : List α) : List α :=
  match l with
  | [] => []
  | x :: y => myReverse y ++ [x]

-- P06 (*): Find out whether a list is a palindrome.
def isPalindrome {α : Type} [BEq α] (l : List α) : Bool :=
  myReverse l == l

-- P07 (**): Flatten a nested list structure.
def myFlatten {α : Type} (nl : NestedList α) : List α :=
  match nl with
  | .elem x => [x]
  | .list x => flattenList x
  where
    flattenList {α : Type} (l : List (NestedList α)) : List α :=
      match l with
      | [] => []
      | y :: ys => myFlatten y ++ flattenList ys

-- P08 (**): Eliminate consecutive duplicates of list elements.
def compress {α : Type} [BEq α] (l : List α) : List α :=
  match l with
  | [] => []
  | [x] => [x]
  | x :: x' :: xs => if x == x' then compress ([x'] ++ xs) else [x] ++ compress ([x'] ++ xs)

-- P09 (**): Pack consecutive duplicates of list elements into sublists.
partial def pack {α : Type} [BEq α] (l : List α) : List (List α) :=
  match l with
  | [] => []
  | x :: xs =>
    match pack xs with
      | (x' :: x's) :: r => if x == x' then (x :: x' :: x's) :: r else [x] :: (x' :: x's) :: r
      | _ => [[x]]

-- P10 (**): Run-length encoding of a list.
def encode {α : Type} [BEq α] (l : List α) : List (Nat × α) :=
  let p := pack l
  help p
  where
    help (l : List (List α)) : List (Nat × α) :=
      match l with
      | [] => []
      | (x :: xs) :: rs => ((myLength xs) + 1, x) :: help rs
      | _ => []

-- P11 (**): Modified run-length encoding.
def encodeModified {α : Type} [BEq α] (l : List α) : List (RLEItem α) :=
  let p := encode l
  help p
  where
    help (l : List (Nat × α)) : List (RLEItem α) :=
      match l with
      | [] => []
      | (a, b) :: rs => if a == 1 then RLEItem.single b :: help rs else RLEItem.run a b :: help rs

-- P12 (**): Decode a run-length encoded list.
def decode {α : Type} [BEq α] (l : List (RLEItem α)) : List α :=
  match l with
  | [] => []
  | RLEItem.single x :: xs => x :: decode xs
  | RLEItem.run n x :: xs => List.replicate n x ++ decode xs

-- P13 (**): Run-length encoding of a list (direct solution).
def encodeDirect {α : Type} [BEq α] (l : List α) : List (RLEItem α) :=
  match l with
  | [] => []
  | x :: xs =>
    match encodeDirect xs with
    | RLEItem.single y :: ys => if x == y then RLEItem.run 2 y :: ys else RLEItem.single x :: RLEItem.single y :: ys
    | RLEItem.run n y :: ys => if x == y then RLEItem.run (n+1) y :: ys else RLEItem.single x :: RLEItem.run n y :: ys
    | _ => [RLEItem.single x]

-- P14 (*): Duplicate the elements of a list.
def dupli {α : Type} (l : List α) : List α :=
  match l with
  | [] => []
  | x :: xs => x :: x :: dupli xs

-- P15 (**): Replicate the elements of a list a given number of times.
def repli {α : Type} (l : List α) (n : Nat) : List α :=
  match l, n with
  | [], _ => []
  | _, 0 => []
  | x :: xs, c => List.replicate c x ++ repli xs c

-- P16 (**): Drop every N'th element from a list (1-indexed).
def dropEvery {α : Type} (l : List α) (n : Nat) : List α :=
  match l, n with
  | xs, 0 => xs
  | _, _ => help l n
  where
    help (l : List α) (c : Nat) : List α :=
      match l, c with
      | [], _ => []
      | _ :: xs, 1 => help xs n
      | x :: xs, cs => x :: help xs (cs-1)

-- P17 (*): Split a list into two parts; the length of the first part is given.
def split {α : Type} (l : List α) (n : Nat) : List α × List α :=
  match l, n with
  | [], _ => ([], [])
  | _, 0 => ([], l)
  | x :: xs, c =>
    match split xs (c-1) with
    | (l1, l2)=> (x :: l1, l2)

-- P18 (**): Extract a slice from a list (1-indexed, s and e inclusive).
def slice {α : Type} (l : List α) (s e : Nat) : List α :=
  match l, s, e with
  | [], _, _ => []
  | x :: _, 1, 1 => [x]
  | x :: xs, 1, ec => x :: slice xs 1 (ec-1)
  | _ :: xs, sc, ec => slice xs (sc-1) (ec-1)

-- P19 (**): Rotate a list N places to the left.
def rotate {α : Type} (l : List α) (n : Int) : List α :=
  let ind : Nat := if n >= 0 then n.toNat else ((myLength l) + n).toNat
  let (f, s) := split l ind
  s ++ f

-- P20 (*): Remove the K'th element from a list (1-indexed).
def removeAt {α : Type} (l : List α) (n : Nat) : List α × Option α :=
  match l, n with
  | [], _ => ([], none)
  | x :: xs, 1 => (xs, x)
  | x :: xs, c =>
    let (rs, r) := removeAt xs (c-1)
    (x :: rs, r)

-- P21 (*): Insert an element at a given position into a list (1-indexed).
def insertAt {α : Type} (e : α) (l : List α) (i : Nat) : List α :=
  match l, i with
  | [], _ => [e]
  | xs, 1 => e :: xs
  | x :: xs, c => x :: insertAt e xs (c-1)

-- P22 (*): Create a list containing all integers within a given range.
partial def myRange (s e : Int) : List Int :=
  if s == e then [s] else
    if s > e then
      myReverse (myRange e s)
    else
      s :: myRange (s+1) e

-- P23 (**): Extract a given number of randomly selected elements from a list.
partial def rndSelect {α : Type} (l : List α) (x : Nat) : IO (List α) := do
  match l, x with
  | _, 0 => pure []
  | [], _ => pure []
  | _, c =>
    let idx ← IO.rand 1 (myLength l)
    let (left, choice) := removeAt l idx
    let rs ← rndSelect left (c-1)
    match choice with
    | some val => pure (val :: rs)
    | _ => pure rs

-- P24 (*): Lotto: Draw N different random numbers from the set 1..M.
partial def lottoSelect (x y : Nat) : IO (List Nat) :=
  rndSelect ((myRange 1 y).map Int.toNat) x

-- P25 (*): Generate a random permutation of the elements of a list.
partial def rndPermu {α : Type} (l : List α) : IO (List α) :=
  rndSelect l (myLength l)

end A1
