use vstd::prelude::*;

verus! {
    fn sum(N: u32) -> (s: u32)
        requires N >= 0,
        ensures s == N * (N + 1) / 2,
    {
        let mut n: u32 = 0;
        let mut s: u32 = 0;
        while n != N
            invariant 
                0 <= n <= N,
                s == n * (n + 1) / 2,
            decreases N - n,
        {
            n = n + 1;
            s = s + n;
        }
        s
    }
}

fn main() {}