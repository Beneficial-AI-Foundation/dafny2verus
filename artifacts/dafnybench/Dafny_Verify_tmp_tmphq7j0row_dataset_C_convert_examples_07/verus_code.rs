use vstd::prelude::*;

verus! {
    // Original Dafny method translated to Verus
    fn main_method(n: usize, choices: Vec<bool>) -> (ret: (usize, usize))
        requires 
            n >= 0,
            choices.len() == n,
            n <= 1000, // bound to prevent overflow
        ensures ret.0 + ret.1 == 3 * n,
    {
        let mut i: usize = 0;
        let mut a: usize = 0;
        let mut b: usize = 0;

        while i < n
            invariant 
                0 <= i <= n,
                a + b == 3 * i,
                choices.len() == n,
                a <= 2 * i,
                b <= 2 * i,
                n <= 1000,
            decreases n - i,
        {
            // Non-deterministic choice represented by the choices vector
            // In Dafny this was: if(*) 
            if choices[i] {
                a = a + 1;
                b = b + 2;
            } else {
                a = a + 2;
                b = b + 1;
            }

            i = i + 1;
        }

        (a, b)
    }

    fn main() {}
}