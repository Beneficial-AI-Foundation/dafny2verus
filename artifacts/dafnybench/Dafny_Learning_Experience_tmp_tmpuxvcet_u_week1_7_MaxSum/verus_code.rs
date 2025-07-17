use vstd::prelude::*;

verus! {
    // Ghost/spec version using abstract int
    spec fn max_sum_spec(x: int, y: int) -> (int, int) {
        let s = x + y;
        let m = if x > y { x } else if y > x { y } else { x };
        (s, m)
    }

    // Executable version using concrete types
    fn max_sum(x: i64, y: i64) -> (r: (i64, i64))
        requires
            x.checked_add(y).is_some(),
        ensures
            r.0 == x + y,
            (r.1 == x || r.1 == y) && x <= r.1 && y <= r.1,
    {
        let s = x + y;
        let m = if x > y {
            x
        } else if y > x {
            y
        } else {
            x
        };
        assert(m >= y);
        (s, m)
    }

    // Proof function to test the specification
    proof fn test_max_sum() {
        let m: int = 4;
        let n: int = 5;
        let (a, b) = max_sum_spec(m, n);
        assert(a == 9);
        assert(b == 5);
    }
}

fn main() {
    let m: i64 = 4;
    let n: i64 = 5;
    let (a, b) = max_sum(m, n);
    println!("Search return a is {}, b is {}", a, b);
}