use vstd::prelude::*;

verus! {
    spec fn Triple(x: int) -> int
    {
        let y = x * 2;
        y + x
    }
    
    proof fn Triple_correctness(x: int) 
        ensures Triple(x) == 3 * x
    {
        assert(Triple(x) == x * 2 + 2);
        assert(x * 2 + x == 3 * x);
    }
}