use vstd::prelude::*;

verus! {
    fn main_method(x: i32) -> (result: (i32, i32))
        requires 
            x > 0,
            x <= 1073741823, // prevent overflow (2^30 - 1)
        ensures result.0 == 2 * x,
    {
        let mut i: i32 = 0;
        let mut j: i32 = 0;

        while i < x
            invariant 
                0 <= i <= x,
                j == 2 * i,
                x <= 1073741823,
            decreases x - i,
        {
            j = j + 2;
            i = i + 1;
            assert(j == 2 * i);
        }

        (j, i)
    }
}

fn main() {}