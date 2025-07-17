use vstd::prelude::*;

verus! {
    spec fn sum_ints(n: int) -> int
        decreases n
    {
        if n <= 0 {
            0
        } else {
            sum_ints(n - 1) + n
        }
    }

    fn sum_ints_loop(n: u32) -> (s: u32)
        requires n >= 0,
        ensures s == sum_ints(n as int),
        ensures s == n * (n + 1) / 2,
    {
        let mut s = 0u32;
        let mut k = 0u32;
        
        while k < n
            invariant 
                0 <= k <= n,
                s == sum_ints(k as int),
                s == k * (k + 1) / 2,
            decreases n - k,
        {
            k = k + 1;
            s = s + k;
        }
        
        s
    }

    fn main() {
        let x = sum_ints_loop(100);
        // Result computed successfully  
    }
}