use vstd::prelude::*;

verus! {
    fn compute_double(x: i32) -> (result: (i32, i32))
        requires 
            x > 0,
            x <= i32::MAX / 2,  // Prevent overflow
        ensures result.0 == 2 * x,
    {
        let mut i = 0i32;
        let mut j = 0i32;

        while i < x
            invariant 
                0 <= i <= x,
                j == 2 * i,
                x <= i32::MAX / 2,
            decreases x - i,
        {
            j = j + 2i32;
            i = i + 1i32;
        }

        (j, i)
    }
}

fn main() {}