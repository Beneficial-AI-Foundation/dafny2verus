use vstd::prelude::*;

verus! {
    fn max(a: i32, b: i32) -> (c: i32)
        ensures c >= a && c >= b
    {
        if a < b {
            assert(a <= b && b <= b);
            b
        } else {
            assert(a <= a && b <= a);
            a
        }
    }

    fn testing() {
        let v = max(2, 3);
        assert(v >= 3);
    }
}

fn main() {}