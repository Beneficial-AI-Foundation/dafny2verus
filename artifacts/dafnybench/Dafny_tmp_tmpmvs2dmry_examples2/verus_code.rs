use vstd::prelude::*;

verus! {
    // Addition by increment - working version
    fn add_by_inc(x: u32, y: u32) -> (z: u32)
        requires x as u64 + y as u64 <= u32::MAX as u64,
        ensures z as u64 == x as u64 + y as u64,
    {
        let mut z = x;
        let mut i = 0u32;
        while i < y
            invariant 
                i <= y,
                z as u64 == x as u64 + i as u64,
                z as u64 <= u32::MAX as u64,
            decreases y - i,
        {
            z = z + 1;
            i = i + 1;
        }
        z
    }

    // Product function - multiplication by repeated addition
    fn product(m: u32, n: u32) -> (res: u32)
        requires m as u64 * n as u64 <= u32::MAX as u64,
        ensures res as u64 == m as u64 * n as u64,
    {
        let mut m1 = m;
        let mut res = 0u32;
        while m1 != 0
            invariant 
                m1 <= m,
                res as u64 == (m - m1) as u64 * n as u64,
                res as u64 <= u32::MAX as u64,
            decreases m1,
        {
            let mut n1 = n;
            while n1 != 0
                invariant 
                    n1 <= n,
                    res as u64 == (m - m1) as u64 * n as u64 + (n - n1) as u64,
                    res as u64 <= u32::MAX as u64,
                decreases n1,
            {
                res = res + 1;
                n1 = n1 - 1;
            }
            m1 = m1 - 1;
        }
        res
    }

    // GCD calculation function (Euclidean algorithm)
    fn gcd_calc(m: u32, n: u32) -> (res: u32)
        requires m > 0 && n > 0,
        ensures res > 0,
    {
        let mut m1 = m;
        let mut n1 = n;
        while m1 != n1
            invariant 
                m1 > 0,
                n1 > 0,
            decreases m1 + n1,
        {
            if m1 > n1 {
                m1 = m1 - n1;
            } else {
                n1 = n1 - m1;
            }
        }
        n1
    }

    // GCD specification function
    spec fn gcd(m: int, n: int) -> int
        recommends m > 0 && n > 0,
        decreases m + n,
    {
        if m == n {
            n
        } else if m > n {
            gcd(m - n, n)
        } else {
            gcd(m, n - m)
        }
    }

    fn main() {}
}