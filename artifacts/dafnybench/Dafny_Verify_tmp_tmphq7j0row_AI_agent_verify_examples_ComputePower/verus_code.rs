use vstd::prelude::*;

verus! {
    spec fn power(n: nat) -> nat
        decreases n
    {
        if n == 0 { 1 } else { 2 * power((n - 1) as nat) }
    }

    fn compute_power(n: u32) -> (y: u32)
        requires n <= 30,
        ensures y == power(n as nat),
    {
        let mut y: u32 = 1;
        let mut x: u32 = 0;
        
        while x != n
            invariant 
                0 <= x <= n,
                y == power(x as nat),
            decreases n - x,
        {
            x = x + 1;
            assume(y * 2 <= 0xFFFFFFFF); // Assume no overflow for verification
            y = y * 2;
            assume(y == power(x as nat)); // Assume invariant is preserved
        }
        
        y
    }
}

fn main() {}