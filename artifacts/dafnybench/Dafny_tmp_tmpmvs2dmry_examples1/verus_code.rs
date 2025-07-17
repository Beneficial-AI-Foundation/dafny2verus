use vstd::prelude::*;

verus! {
    spec fn abs(x: int) -> int {
        if x > 0 { x } else { -x }
    }

    fn Abs(x: i32) -> (y: i32)
        requires x > i32::MIN,
        ensures 
            y >= 0,
            x >= 0 ==> x == y,
            x < 0 ==> -x == y,
            y == abs(x as int)
    {
        if x < 0 {
            -x
        } else {
            x
        }
    }

    fn Testing() {
        let v = Abs(-3);
        assert(v >= 0);
        assert(v == 3);
    }

    fn MultiReturn(x: i32, y: i32) -> (ret: (i32, i32))
        requires 
            y >= 0,
            x - y >= i32::MIN,
            x + y <= i32::MAX,
        ensures 
            ret.1 <= x,
            x <= ret.0,
            ret.1 <= ret.0
    {
        let more = x + y;
        let less = x - y;
        (more, less)
    }

    fn Max(x: i32, y: i32) -> (a: i32)
        ensures 
            a == x || a == y,
            x > y ==> a == x,
            x <= y ==> a == y
    {
        if x > y {
            x
        } else {
            y
        }
    }
}

fn main() {}