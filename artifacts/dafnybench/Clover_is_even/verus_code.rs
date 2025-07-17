use vstd::prelude::*;

verus! {
    fn compute_is_even(x: u32) -> (is_even: bool)
        ensures (x % 2 == 0) == is_even
    {
        let mut is_even = false;
        if x % 2 == 0 {
            is_even = true;
        }
        is_even
    }
}

fn main() {}