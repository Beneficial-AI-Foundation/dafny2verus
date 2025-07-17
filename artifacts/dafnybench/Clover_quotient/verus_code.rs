use vstd::prelude::*;

verus! {
    fn quotient(x: u32, y: u32) -> (result: (u32, u32))
        requires 
            y > 0,
            x < 1000,
            y < 1000,
        ensures 
            result.0 * y + result.1 == x,
            result.1 < y,
    {
        let mut r: u32 = x;
        let mut q: u32 = 0;
        
        while y <= r
            invariant 
                q * y + r == x,
                q < 1000,
            decreases r,
        {
            r = r - y;
            q = q + 1;
        }
        
        (q, r)
    }
}

fn main() {
    let result = quotient(10, 3);
    println!("quotient(10, 3) = {:?}", result);
}