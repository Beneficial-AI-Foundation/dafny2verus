use vstd::prelude::*;

verus! {
    // Direct translation from Dafny method to Verus spec function
    spec fn triple(x: int) -> int {
        if x < 18 {
            let a = 2 * x;
            let b = 4 * x;
            (a + b) / 2
        } else {
            // In original Dafny: "case 0 <= x"
            // Since conditions can overlap, we use else here
            let y = 2 * x;
            x + y
        }
    }
    
    // Proof that the function correctly computes 3*x
    proof fn triple_correctness(x: int) 
        ensures triple(x) == 3 * x
    {
        if x < 18 {
            // First branch: (2*x + 4*x) / 2 = 6*x / 2 = 3*x
            let a = 2 * x;
            let b = 4 * x;
            assert((a + b) / 2 == (2 * x + 4 * x) / 2);
            assert((2 * x + 4 * x) / 2 == (6 * x) / 2);
            assert((6 * x) / 2 == 3 * x);
        } else {
            // Second branch: x + 2*x = 3*x
            let y = 2 * x;
            assert(x + y == x + 2 * x);
            assert(x + 2 * x == 3 * x);
        }
    }
}