use vstd::prelude::*;

verus! {
    fn sum_loop(n: u64) -> (ret: (u64, u64))
        requires 
            n >= 0,
            n <= 10  // Small upper bound to prevent overflow
        ensures 
            ret.0 >= 0,
            0 <= ret.1 <= n,
            ret.1 == n
    {
        let mut sum = 0u64;
        let mut i = 0u64;
        
        while i < n
            invariant 
                0 <= i <= n,
                n <= 10,
                sum <= i * 10  // Since each iteration adds at most i and i <= n <= 10
            decreases n - i
        {
            sum = sum + i;
            i = i + 1u64;
        }
        
        (sum, i)
    }
}

fn main() {}