use vstd::prelude::*;

verus! {
    fn cal_div() -> (ret: (i32, i32))
        ensures
            ret.0 == 191i32 / 7i32,
            ret.1 == 191i32 % 7i32,
    {
        let mut x = 0i32;
        let mut y = 191i32;
        
        while 7i32 <= y
            invariant
                0i32 <= y,
                7i32 * x + y == 191i32,
            decreases y,
        {
            x = x + 1i32;
            y = 191i32 - 7i32 * x;
        }
        
        (x, y)
    }
}