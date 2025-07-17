use vstd::prelude::*;

verus! {
    fn main_method() -> (ret: (i32, i32, i32, i32))
        ensures ret.3 >= 1
    {
        let mut x: i32 = 1;
        let mut y: i32 = 1;
        let mut t1: i32 = 0;
        let mut t2: i32 = 0;

        while x <= 100000
            invariant 
                x == y,
                x >= 1,
                y >= 1,
                x <= 200000  // prevent overflow
            decreases 200000 - x
        {
            t1 = x;
            t2 = y;
            x = t1 + t2;
            y = t1 + t2;
        }

        (t1, t2, x, y)
    }
}

fn main() {}