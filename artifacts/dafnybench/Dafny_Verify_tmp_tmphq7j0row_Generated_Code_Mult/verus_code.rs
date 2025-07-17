use vstd::prelude::*;

verus! {
    fn mult(a: u32, b: u32) -> (x: u32)
        requires 
            a >= 0 && b >= 0,
            a <= 100,
            b <= 100,
        ensures x == a * b,
    {
        let mut x: u32 = 0;
        let mut y: u32 = a;
        while y > 0
            invariant 
                x == (a - y) * b,
                y <= a,
            decreases y,
        {
            x = x + b;
            y = y - 1;
        }
        x
    }
}