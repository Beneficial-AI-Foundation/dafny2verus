use vstd::prelude::*;

verus! {
    fn cube(n: usize) -> (c: usize)
        ensures c == n * n * n
    {
        let mut c: usize = 0;
        let mut i: usize = 0;
        let mut k: usize = 1;
        let mut m: usize = 6;
        
        while i != n
            invariant 
                0 <= i <= n,
                c == i * i * i,
                k == 3 * i * i + 3 * i + 1,
                m == 6 * i + 6
            decreases n - i
        {
            c = c + k;
            k = k + m;
            m = m + 6;
            i = i + 1;
        }
        
        c
    }
}