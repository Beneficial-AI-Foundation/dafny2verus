use vstd::prelude::*;

verus! {
    spec fn Average(a: int, b: int) -> int {
        (a + b) / 2
    }

    proof fn Triple(x: int) -> (r: int)
        ensures r == 3 * x
    {
        let r = Average(2 * x, 4 * x);
        assert((2 * x + 4 * x) / 2 == 6 * x / 2);
        assert(6 * x / 2 == 3 * x);
        r
    }

    fn Triple1(x: i32) -> (r: i32)
        requires -715827882 <= x <= 715827882  // Prevent overflow
        ensures r == 3 * x
    {
        let y = 2 * x;
        let r = x + y;
        
        proof {
            let ghost (a, b) = DoubleQuadruple(x as int);
            // Since r = x + 2*x = 3*x, a = 2*x, b = 4*x
            // We have a <= r <= b when x >= 0, and b <= r <= a when x < 0
            assert(a <= (r as int) <= b || b <= (r as int) <= a);
        }
        
        r
    }

    proof fn DoubleQuadruple(x: int) -> (res: (int, int))
        ensures res.0 == 2 * x && res.1 == 4 * x
    {
        let a = 2 * x;
        let b = 2 * a;
        (a, b)
    }

    fn F() -> (r: i32)
        ensures r == 29
    {
        29
    }

    fn M() -> (r: i32)
        ensures r == 29
    {
        29
    }

    fn Caller() {
        let a = F();
        let b = M();
        assert(a == 29);
        assert(b == 29);
    }

    fn MyMethod(x: i32) -> (y: i32)
        requires 10 <= x <= 1000  // Prevent overflow and ensure postcondition
        ensures 25 <= y
    {
        let a = x + 3;

        let b = if x < 20 {
            32 - x
        } else {
            16
        };

        let y = a + b;
        
        // Proof that y >= 25
        assert(a == x + 3);
        if x < 20 {
            assert(b == 32 - x);
            assert(y == (x + 3) + (32 - x));
            assert(y == 35);
        } else {
            assert(b == 16);
            assert(y == (x + 3) + 16);
            assert(y == x + 19);
            assert(x >= 20);
            assert(y >= 20 + 19);
            assert(y >= 39);
        }
        assert(y >= 25);
        
        y
    }
}

fn main() {}