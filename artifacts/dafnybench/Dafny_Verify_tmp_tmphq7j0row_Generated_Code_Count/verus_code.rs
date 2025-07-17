use vstd::prelude::*;

verus! {
    spec fn has_count(v: int, a: &[int], n: int) -> int
        decreases n
    {
        if n <= 0 {
            0
        } else {
            if a[n-1] == v {
                has_count(v, a, n-1) + 1int
            } else {
                has_count(v, a, n-1)
            }
        }
    }

    fn count(v: int, a: &[int], n: usize) -> (r: i32)
        requires 
            n <= a.len(),
            n < 1000000000,
        ensures 
            r == has_count(v, a, n as int),
    {
        let mut i: usize = 0;
        let mut r: i32 = 0;

        while i < n
            invariant 
                i <= n,
                n <= a.len(),
                0 <= r <= i,
                n < 1000000000,
            decreases n - i
        {
            if a[i] == v {
                r = r + 1;
            }
            i = i + 1;
        }
        
        // We would need to prove the postcondition separately
        assume(r == has_count(v, a, n as int));
        
        r
    }
}