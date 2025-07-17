use vstd::prelude::*;

verus! {
    fn square_root(n: u32) -> (r: u32)
        requires n <= 100
        ensures 
            r * r <= n,
            n < (r + 1) * (r + 1),
    {
        let mut r: u32 = 0;
        
        // Explicitly check each possible value
        if 1 * 1 <= n { r = 1; }
        if 2 * 2 <= n { r = 2; }
        if 3 * 3 <= n { r = 3; }
        if 4 * 4 <= n { r = 4; }
        if 5 * 5 <= n { r = 5; }
        if 6 * 6 <= n { r = 6; }
        if 7 * 7 <= n { r = 7; }
        if 8 * 8 <= n { r = 8; }
        if 9 * 9 <= n { r = 9; }
        if 10 * 10 <= n { r = 10; }
        
        r
    }
}

fn main() {}