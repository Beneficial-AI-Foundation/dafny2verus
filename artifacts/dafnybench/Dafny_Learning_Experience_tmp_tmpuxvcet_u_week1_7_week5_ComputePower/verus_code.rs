use vstd::prelude::*;

verus! {
    spec fn power(n: nat) -> nat
        decreases n
    {
        if n == 0 { 1 } else { 2 * power((n - 1) as nat) }
    }

    fn calc_power(n: u32) -> (p: u32)
        ensures p as nat == 2 * n
    {
        2 * n
    }

    fn compute_power(n: u32) -> (p: u32)
        ensures p as nat == power(n as nat)
    {
        let mut p: u32 = 1;
        let mut i: u32 = 0;
        while i != n
            invariant 
                0 <= i <= n,
                p as nat * power((n - i) as nat) == power(n as nat)
            decreases n - i
        {
            p = calc_power(p);
            i = i + 1;
        }
        p
    }
}

fn main() {}