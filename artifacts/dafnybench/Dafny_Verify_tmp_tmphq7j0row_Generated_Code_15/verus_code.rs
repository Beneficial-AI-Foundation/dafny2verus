use vstd::prelude::*;

verus! {
    fn compute_k(n: i32, k: i32) -> (k_out: i32)
        requires 
            n > 0,
            k >= n + 1, 
            k >= 0,
            n <= 1000000,  
            k <= 1000000,
        ensures k_out >= 0,
    {
        let mut k_out = k;
        let mut j: i32 = 0;
        while j < n
            invariant 
                0 <= j <= n,
                k_out == k - j,
                k_out >= k - n,
                k >= n + 1,  // Need this to ensure no underflow
            decreases n - j,
        {
            j = j + 1;
            k_out = k_out - 1;
        }
        k_out
    }
}

fn main() {}