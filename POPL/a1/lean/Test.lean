-- Your own tests for the Lean component, the counterpart of racket/test.rkt.
--
--   cd lean
--   lake build A1        -- compile your solutions
--   lake env lean Test.lean
--
-- `#guard` reports an error unless the expression evaluates to `true`, so a
-- file that elaborates without complaint is a file whose tests all passed.

import A1

open A1

-- P01: myLast
#guard myLast ["a", "b", "c", "d"] == some "d"
#guard myLast [1, 2, 3] == some 3
#guard myLast [42] == some 42
#guard myLast ([] : List Int) == none

-- P02: myButLast
#guard myButLast ["a", "b", "c", "d"] == some ["c", "d"]
#guard myButLast [1, 2] == some [1, 2]
#guard myButLast [1] == some [1]
#guard myButLast ([] : List Int) == none

-- P03: elementAt
#guard elementAt ["a", "b", "c", "d", "e"] 3 == some "c"
#guard elementAt [10, 20, 30] 1 == some 10
#guard elementAt [10, 20, 30] 0 == none
#guard elementAt [10, 20, 30] 4 == none
#guard elementAt ([] : List Int) 1 == none

-- P04: myLength
#guard myLength ["a", "b", "c"] == 3
#guard myLength [1, 2, 3, 4, 5] == 5
#guard myLength ([] : List Int) == 0

-- P05: myReverse
#guard myReverse ["a", "b", "c"] == ["c", "b", "a"]
#guard myReverse [1, 2, 3, 4] == [4, 3, 2, 1]
#guard myReverse ([] : List Int) == []

-- P06: isPalindrome
#guard isPalindrome ["x", "a", "m", "a", "x"] == true
#guard isPalindrome [1, 2, 3, 2, 1] == true
#guard isPalindrome [1, 2, 3, 4] == false
#guard isPalindrome ([] : List Int) == true

-- P07: myFlatten
#guard myFlatten (NestedList.elem 1) == [1]
#guard myFlatten (NestedList.list [NestedList.elem "a", NestedList.list [NestedList.elem "b", NestedList.list [NestedList.elem "c", NestedList.elem "d"]], NestedList.elem "e"]) == ["a", "b", "c", "d", "e"]
#guard myFlatten (NestedList.list [] : NestedList Int) == []

-- P08: compress
#guard compress ["a", "a", "a", "a", "b", "c", "c", "a", "a", "d", "e", "e", "e", "e"] == ["a", "b", "c", "a", "d", "e"]
#guard compress [1, 1, 1, 2, 2, 3] == [1, 2, 3]
#guard compress [1, 2, 3] == [1, 2, 3]
#guard compress ([] : List Int) == []

-- P09: pack
#guard pack ["a", "a", "a", "a", "b", "c", "c", "a", "a", "d", "e", "e", "e", "e"] == [["a", "a", "a", "a"], ["b"], ["c", "c"], ["a", "a"], ["d"], ["e", "e", "e", "e"]]
#guard pack [1, 1, 2, 3, 3, 3] == [[1, 1], [2], [3, 3, 3]]
#guard pack ([] : List Int) == []

-- P10: encode
#guard encode ["a", "a", "a", "a", "b", "c", "c", "a", "a", "d", "e", "e", "e", "e"] == [(4, "a"), (1, "b"), (2, "c"), (2, "a"), (1, "d"), (4, "e")]
#guard encode [1, 1, 2, 3, 3, 3] == [(2, 1), (1, 2), (3, 3)]
#guard encode ([] : List Int) == []

-- P11: encodeModified
#guard encodeModified ["a", "a", "a", "a", "b", "c", "c", "a", "a", "d", "e", "e", "e", "e"] == [RLEItem.run 4 "a", RLEItem.single "b", RLEItem.run 2 "c", RLEItem.run 2 "a", RLEItem.single "d", RLEItem.run 4 "e"]
#guard encodeModified [1, 2, 2, 3] == [RLEItem.single 1, RLEItem.run 2 2, RLEItem.single 3]
#guard encodeModified ([] : List Int) == []

-- P12: decode
#guard decode [RLEItem.run 4 "a", RLEItem.single "b", RLEItem.run 2 "c", RLEItem.run 2 "a", RLEItem.single "d", RLEItem.run 4 "e"] == ["a", "a", "a", "a", "b", "c", "c", "a", "a", "d", "e", "e", "e", "e"]
#guard decode [RLEItem.single 1, RLEItem.run 2 2, RLEItem.single 3] == [1, 2, 2, 3]
#guard decode ([] : List (RLEItem Int)) == []

-- P13: encodeDirect
#guard encodeDirect ["a", "a", "a", "a", "b", "c", "c", "a", "a", "d", "e", "e", "e", "e"] == [RLEItem.run 4 "a", RLEItem.single "b", RLEItem.run 2 "c", RLEItem.run 2 "a", RLEItem.single "d", RLEItem.run 4 "e"]
#guard encodeDirect [1, 2, 2, 3] == [RLEItem.single 1, RLEItem.run 2 2, RLEItem.single 3]
#guard encodeDirect ([] : List Int) == []

-- P14: dupli
#guard dupli ["a", "b", "c", "c", "d"] == ["a", "a", "b", "b", "c", "c", "c", "c", "d", "d"]
#guard dupli [1, 2, 3] == [1, 1, 2, 2, 3, 3]
#guard dupli ([] : List Int) == []

-- P15: repli
#guard repli ["a", "b", "c"] 3 == ["a", "a", "a", "b", "b", "b", "c", "c", "c"]
#guard repli [1, 2] 0 == []
#guard repli [1, 2] 1 == [1, 2]
#guard repli ([] : List Int) 3 == []

-- P16: dropEvery
#guard dropEvery ["a", "b", "c", "d", "e", "f", "g", "h", "i", "k"] 3 == ["a", "b", "d", "e", "g", "h", "k"]
#guard dropEvery [1, 2, 3, 4, 5, 6] 2 == [1, 3, 5]
#guard dropEvery [1, 2, 3] 0 == [1, 2, 3]
#guard dropEvery ([] : List Int) 2 == []

-- P17: split
#guard split ["a", "b", "c", "d", "e", "f", "g", "h", "i", "k"] 3 == (["a", "b", "c"], ["d", "e", "f", "g", "h", "i", "k"])
#guard split [1, 2, 3] 0 == ([], [1, 2, 3])
#guard split [1, 2, 3] 5 == ([1, 2, 3], [])
#guard split ([] : List Int) 2 == ([], [])

-- P18: slice
#guard slice ["a", "b", "c", "d", "e", "f", "g", "h", "i", "k"] 3 7 == ["c", "d", "e", "f", "g"]
#guard slice [10, 20, 30, 40, 50] 2 4 == [20, 30, 40]
#guard slice [10, 20, 30] 1 1 == [10]
#guard slice ([] : List Int) 1 3 == []

-- P19: rotate
#guard rotate ["a", "b", "c", "d", "e", "f", "g", "h"] 3 == ["d", "e", "f", "g", "h", "a", "b", "c"]
#guard rotate ["a", "b", "c", "d", "e", "f", "g", "h"] (-2) == ["g", "h", "a", "b", "c", "d", "e", "f"]
#guard rotate [1, 2, 3, 4, 5] 0 == [1, 2, 3, 4, 5]
#guard rotate ([] : List Int) 2 == []

-- P20: removeAt
#guard removeAt ["a", "b", "c", "d"] 2 == (["a", "c", "d"], some "b")
#guard removeAt [10, 20, 30] 1 == ([20, 30], some 10)
#guard removeAt [10, 20, 30] 0 == ([10, 20, 30], none)
#guard removeAt [10, 20, 30] 5 == ([10, 20, 30], none)
#guard removeAt ([] : List Int) 1 == ([], none)

-- P21: insertAt
#guard insertAt "alfa" ["a", "b", "c", "d"] 2 == ["a", "alfa", "b", "c", "d"]
#guard insertAt 99 [10, 20, 30] 1 == [99, 10, 20, 30]
#guard insertAt 99 [10, 20, 30] 4 == [10, 20, 30, 99]
#guard insertAt 99 ([] : List Int) 1 == [99]

-- P22: myRange
#guard myRange 4 9 == [4, 5, 6, 7, 8, 9]
#guard myRange 9 4 == [9, 8, 7, 6, 5, 4]
#guard myRange 5 5 == [5]

-- P23: rndSelect
#eval (do
  let res ← rndSelect ["a", "b", "c", "d", "e", "f"] 3
  if res.length == 3 then
    pure ()
  else
    throw (IO.userError "rndSelect test failed")
)

-- P24: lottoSelect
#eval (do
  let res ← lottoSelect 6 49
  if res.length == 6 then
    pure ()
  else
    throw (IO.userError "lottoSelect test failed")
)

-- P25: rndPermu
#eval (do
  let res ← rndPermu ["a", "b", "c", "d"]
  if res.length == 4 then
    pure ()
  else
    throw (IO.userError "rndPermu test failed")
)
