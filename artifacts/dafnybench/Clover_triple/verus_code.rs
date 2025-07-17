use vstd::prelude::*;

verus! {
    fn triple(x: i64) -> (r: i64)
        requires -1000000 <= x <= 1000000, // prevent overflow
        ensures r == 3 * x
    {
        x * 3
    }
}

fn main() {}