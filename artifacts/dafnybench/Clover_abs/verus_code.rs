use vstd::prelude::*;

verus! {
    fn abs(x: i32) -> (y: i32)
        requires x > i32::MIN,
        ensures
            x >= 0 ==> x == y,
            x < 0 ==> x + y == 0,
    {
        if x < 0 {
            -x
        } else {
            x
        }
    }
}