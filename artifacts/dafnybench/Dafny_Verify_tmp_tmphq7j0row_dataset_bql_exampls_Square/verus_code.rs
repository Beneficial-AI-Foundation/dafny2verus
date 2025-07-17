use vstd::prelude::*;

verus! {
    fn square(n: u32) -> (r: u32)
        requires n <= 46340,
        ensures r == n * n,
    {
        let mut x: u32 = 1;
        let mut i: u32 = 0;
        let mut r: u32 = 0;

        while i < n
            invariant 
                i <= n,
                r == i * i,
                x == 2 * i + 1,
            decreases n - i,
        {
            // The key mathematical insight: r + x = i^2 + (2i + 1) = (i + 1)^2
            assert(r + x == (i + 1) * (i + 1)) by {
                assert(r == i * i);
                assert(x == 2 * i + 1);
                assert(r + x == i * i + 2 * i + 1);
                assert(i * i + 2 * i + 1 == (i + 1) * (i + 1)) by (nonlinear_arith);
            }
            
            r = r + x;
            i = i + 1;
            x = x + 2;
        }
        
        r
    }
}

#[verifier::external]
fn main() {
    println!("Square function translated from Dafny to Verus");
}