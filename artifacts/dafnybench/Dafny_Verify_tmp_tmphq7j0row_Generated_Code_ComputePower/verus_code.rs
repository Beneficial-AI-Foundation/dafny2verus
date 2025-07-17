use vstd::prelude::*;

verus! {
    spec fn power(n: nat) -> nat
        decreases n
    {
        if n == 0 { 1 } else { 2 * power(sub(n, 1)) }
    }

    fn compute_power(n: u32) -> (p: u32)
        requires n <= 20,  // Bound to prevent overflow
        ensures p == power(n as nat)
    {
        let mut p: u32 = 1;
        let mut i: u32 = 0;
        
        while i != n
            invariant 
                0 <= i <= n,
                p == power(i as nat),
                n <= 20,
                i <= 20,
            decreases n - i
        {
            i = i + 1;
            p = p * 2;
        }
        p
    }
}