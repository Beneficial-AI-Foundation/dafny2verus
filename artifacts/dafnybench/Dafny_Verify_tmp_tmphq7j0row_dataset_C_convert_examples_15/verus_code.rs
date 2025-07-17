use vstd::prelude::*;

verus! {
    fn main_method(n: i32, k: i32) -> (k_out: i32)
        requires 
            n > 0,
            k > n,
            n <= 100,    
            k <= 200,
        ensures 
            k_out >= 0,
    {
        let mut k_out = k;
        let mut j: i32 = 0;
        
        while j < n
            invariant 
                0 <= j <= n,
                j + k_out == k,
                k_out >= k - n > 0,  // This prevents underflow
            decreases n - j,
        {
            j = j + 1;
            k_out = k_out - 1;
        }
        
        k_out
    }
}

fn main() {}