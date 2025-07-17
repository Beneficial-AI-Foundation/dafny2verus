use vstd::prelude::*;

verus! {
    spec fn even(n: int) -> bool 
        recommends n >= 0
        decreases n
    {
        if n == 0 { 
            true 
        } else if n > 0 { 
            !even(n - 1) 
        } else {
            arbitrary()
        }
    }

    fn is_even(n: u32) -> (r: bool)
        requires n >= 0,
        ensures r <==> even(n as int)
    {
        let mut i: u32 = 0;
        let mut r: bool = true;

        while i < n
            invariant 
                0 <= i <= n,
                r <==> even(i as int)
            decreases n - i
        {
            r = !r;
            i = i + 1;
        }
        
        r
    }
}

fn main() {}