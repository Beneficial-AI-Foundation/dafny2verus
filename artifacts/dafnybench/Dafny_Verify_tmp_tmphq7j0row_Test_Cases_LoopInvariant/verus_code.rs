use vstd::prelude::*;

verus! {
    fn up_while_less(N: i32) -> (i: i32)
        requires 0 <= N,
        ensures i == N,
    {
        let mut i: i32 = 0;
        while i < N
            invariant 0 <= i <= N,
            decreases N - i,
        {
            i = i + 1;
        }
        i
    }

    fn up_while_not_equal(N: i32) -> (i: i32)
        requires 0 <= N,
        ensures i == N,
    {
        let mut i: i32 = 0;
        while i != N
            invariant 0 <= i <= N,
            decreases N - i,
        {
            i = i + 1;
        }
        i
    }

    fn down_while_not_equal(N: i32) -> (i: i32)
        requires 0 <= N,
        ensures i == 0,
    {
        let mut i: i32 = N;
        while i != 0
            invariant 0 <= i <= N,
            decreases i,
        {
            i = i - 1;
        }
        i
    }

    fn down_while_greater(N: i32) -> (i: i32)
        requires 0 <= N,
        ensures i == 0,
    {
        let mut i: i32 = N;
        while 0 < i
            invariant 0 <= i <= N,
            decreases i,
        {
            i = i - 1;
        }
        i
    }

    fn quotient()
    {
        let mut x: i32 = 0;
        let mut y: i32 = 191;
        while 7 <= y
            invariant 0 <= y && 7 * x + y == 191,
            decreases y,
        {
            y = y - 7;
            x = x + 1;
        }
        assert(x == 191i32 / 7i32 && y == 191i32 % 7i32);
    }

    fn quotient1()
    {
        let mut x: i32 = 0;
        let mut y: i32 = 191;
        while 7 <= y
            invariant 0 <= y && 7 * x + y == 191,
            decreases y,
        {
            x = 27;
            y = 2;
        }
        assert(x == 191i32 / 7i32 && y == 191i32 % 7i32);
    }
}

fn main() {}