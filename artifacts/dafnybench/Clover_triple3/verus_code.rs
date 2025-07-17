use vstd::prelude::*;

verus! {
    fn triple(x: i32) -> (r: i32)
        requires 
            -715827882 <= x <= 715827882,  // i32::MAX / 3 approximately
        ensures r == 3 * x
    {
        if x == 0 {
            0
        } else {
            let y: i32 = 2 * x;
            x + y
        }
    }
}

fn main() {}