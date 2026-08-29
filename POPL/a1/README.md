# Assignment 1: Functional Programming & List Processing

Solve the first 25 problems from [99 Problems in Lisp](https://www.ic.unicamp.br/~meidanis/courses/mc336/problemas-lisp/L-99_Ninety-Nine_Lisp_Problems.html)
twice: once in Racket, once in Lean 4.

```
racket/main.rkt      your Racket solutions          <- graded
racket/test.rkt      your own rackunit tests
lean/A1.lean         your Lean solutions            <- graded
lean/Test.lean       your own Lean tests
```

> [!CAUTION]
> Do not alter any files inside `.github/` or `tests/`.

## Racket

Write your solutions in `racket/main.rkt` and
[`provide`](https://docs.racket-lang.org/guide/module-provide.html) them (the
file already ends with `(provide (all-defined-out))`). Use exactly these names:

`my-last`, `my-but-last`, `element-at`, `my-length`, `my-reverse`, `palindrome?`,
`my-flatten`, `compress`, `pack`, `encode`, `encode-modified`, `decode`,
`encode-direct`, `dupli`, `repli`, `drop`, `split`, `slice`, `rotate`,
`remove-at`, `insert-at`, `my-range`, `rnd-select`, `lotto-select`, `rnd-permu`

Handle standard cases. If a function should return a list, return the empty list
where nothing applies; if it should return an element, return null (the same
thing). Use `racket/random` for randomness.

## Lean 4

Write your solutions in `lean/A1.lean`, inside `namespace A1`, keeping the names
**and type signatures** the skeleton gives you. Those signatures say in the type
what Racket expresses by returning `'()`:

- `myLast`, `elementAt` return `Option α`; `myButLast` returns `Option (List α)`
- `split` returns a pair; `removeAt` returns the rest of the list *and* the removed element
- `myFlatten` takes a `NestedList α`; `encodeModified`, `decode` and `encodeDirect` use `RLEItem α`
- `rndSelect`, `lottoSelect`, `rndPermu` live in `IO` — use `IO.rand`

Compile with `cd lean && lake build A1`.

## What you may not use

The point of the assignment is to write these functions yourself, so the
standard library versions are off limits. Using one costs you that problem.

**Racket** — these are disabled while grading, so calling one produces wrong
answers rather than an error:

`reverse`, `length`, `take`, `takef`, `take-right`, `takef-right`, `drop`,
`drop-right`, `dropf-right`, `split-at`, `split-at-right`, `flatten`,
`list-ref`, `list-set`, `set!`

**Lean** — these are rejected by name:

`List.reverse`, `List.length`, `List.take`, `List.takeWhile`, `List.drop`,
`List.dropWhile`, `List.dropLast`, `List.splitAt`, `List.flatten`, `List.get`,
`List.getD`, indexing (`l[i]`, `getElem`, `getElem?`, `getElem!`), `List.set`,
`List.modify`, `Array.reverse`, `Array.size`, `Array.set`, plus the four that
would each hand you a whole problem: `List.splitBy` (P09), `List.rotateLeft` and
`List.rotateRight` (P19), `List.eraseIdx` (P20), `List.insertIdx` (P21).

It makes no difference *how* you reach them: `l.reverse`, `open List` followed by
`reverse`, and a helper of your own that calls `List.reverse` are all caught, and
a problem that depends on such a helper is penalised too. What you *may* use
freely: `first`/`rest`/`head?`/`tail`, `map`, `filter`, `foldr`, `append`,
`make-list`/`List.replicate`, `range`/`List.range`, and pattern matching.

Everything else is fair game — write as many helpers as you like, and try to
avoid `for`, loops and mutation.

## Testing your own work

```sh
cd racket && racket test.rkt                          # rackunit
cd lean && lake build A1 && lake env lean Test.lean   # #guard
```

```racket
(check-equal? (my-last '(a b c)) 'c "my-last 1")
```

```lean
#guard myLast ["a", "b", "c"] == some "c"
```

`#guard` reports an error unless the expression is `true`, so a `Test.lean` that
elaborates quietly is one whose checks all passed.

## How you are graded

Pushing runs both graders. Each component is worth **134 points**, one point per
check, 268 in total. The Actions log lists every check with a ✅ or ❌, and the
grading step above it explains each failure — for example
`expected (a b c), got (a b)`.

Some mistakes cost more than the checks they fail:

| What you did | What it costs |
| --- | --- |
| Wrong answer | the checks that disagree |
| Used a banned function (Lean) | every check of that problem, and of any problem that uses it |
| Left a `sorry` (Lean) | every check of that problem |
| A function that never terminates | every check of that problem (2 minute limit) |
| Renamed a function, or changed a Lean signature | that whole component scores 0 |
| Code that does not compile | that whole component scores 0 |

A partially finished assignment still earns marks for the problems you did
finish — as long as everything compiles and every function keeps its name.
