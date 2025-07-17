use vstd::prelude::*;

verus! {
    fn main_method(n: usize, choices: Vec<bool>) -> (result: (usize, usize))
        requires 
            n > 0,
            choices.len() >= n,
        ensures ({
            let (x, m) = result;
            (n <= 0) || (0 <= m && m < n)
        })
    {
        let mut x: usize = 0;
        let mut m: usize = 0;

        while x < n
            invariant 
                0 <= x <= n,
                0 <= m < n,
                choices.len() >= n,
            decreases n - x,
        {
            // Non-deterministic choice using the choices sequence
            if choices[x] {
                m = x;
            }
            x = x + 1;
        }
        
        (x, m)
    }
}

fn main() {}