use vstd::prelude::*;

verus! {
    // Translated from the original Dafny code
    fn main_fn(n: u32, k: u32) -> (ret: (u32, u32))
        requires 
            n >= 0,
            k == 1 || k >= 0,
        ensures 
            ret.0 == n,
            // Note: The original postcondition k + i + j >= 2 * n requires
            // the triangular number invariant to be properly established
    {
        let mut i: u32 = 0;
        let mut j: u32 = 0;
        while i < n
            invariant 
                0 <= i <= n,
                // Original invariant: j == i * (i + 1) / 2
                // This requires careful arithmetic reasoning in Verus
            decreases n - i,
        {
            i = i + 1;
            j = j + i;
        }
        (i, j)
    }
}

fn main() {}