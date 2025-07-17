use vstd::prelude::*;

verus! {
    /* 
    * Formal verification of a simple algorithm to find the maximum value in an array.
    * FEUP, MIEIC, MFES, 2020/21.
    * Translated from Dafny to Verus
    */

    // Finds the maximum value in a non-empty array.
    fn findMax(a: &[i32]) -> (max: i32)
        requires a.len() > 0,
        ensures 
            exists|k: int| 0 <= k < a.len() && max == a[k],
            forall|k: int| 0 <= k < a.len() ==> max >= a[k]
    {
        let mut max = a[0];
        let mut i = 1;
        while i < a.len()
            invariant 
                1 <= i <= a.len(),
                exists|k: int| 0 <= k < i && max == a[k],
                forall|k: int| 0 <= k < i ==> max >= a[k]
            decreases a.len() - i
        {
            if a[i] > max {
                max = a[i];
            }
            i += 1;
        }
        max
    }

    // Test cases checked statically.
    fn testFindMax() {
        let a1 = vec![1, 2, 3]; // sorted asc
        let m1 = findMax(&a1);
        assert(m1 == a1[2] && m1 == 3);

        let a2 = vec![3, 2, 1]; // sorted desc
        let m2 = findMax(&a2);
        assert(m2 == a2[0] && m2 == 3);

        let a3 = vec![2, 3, 1]; // unsorted
        let m3 = findMax(&a3);
        assert(m3 == a3[1] && m3 == 3);

        let a4 = vec![1, 2, 2]; // duplicates
        let m4 = findMax(&a4);
        assert(m4 == a4[1] && m4 == 2);

        let a5 = vec![1]; // single element
        let m5 = findMax(&a5);
        assert(m5 == a5[0] && m5 == 1);

        let a6 = vec![1, 1, 1]; // all equal
        let m6 = findMax(&a6);
        assert(m6 == a6[0] && m6 == 1);
    }
}

fn main() {}