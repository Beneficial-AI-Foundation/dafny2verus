use vstd::prelude::*;

verus! {
    fn main_method(n: u32) -> (result: (u32, u32, u32))
        requires n >= 0,
        ensures ({
            let (i, x, y) = result;
            (i % 2 != 0) || (x == 2 * y)
        }),
    {
        let mut i: u32 = 0;
        let mut x: u32 = 0;
        let mut y: u32 = 0;

        while i < n
            invariant 
                0 <= i <= n,
                x == i,
                y == i / 2,
            decreases n - i,
        {
            i = i + 1;
            x = x + 1;
            if i % 2 == 0 {
                y = y + 1;
            }
        }
        
        (i, x, y)
    }
}

fn main() {}