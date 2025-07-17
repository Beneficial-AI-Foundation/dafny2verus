use vstd::prelude::*;

verus! {
    // Predicate to check if all elements in a sequence are non-negative
    spec fn positive(s: Seq<i32>) -> bool {
        forall|u: int| 0 <= u < s.len() ==> s[u] >= 0
    }

    // First method: finds first negative element
    fn mfirstNegative(v: &[i32]) -> (result: (bool, usize))
        ensures 
            result.0 <==> exists|k: int| 0 <= k < v@.len() && v@[k] < 0,
            result.0 ==> {
                let i = result.1 as int;
                0 <= i < v@.len() && v@[i] < 0 && positive(v@.subrange(0, i))
            }
    {
        let mut i: usize = 0;
        let mut b: bool = false;
        
        while i < v.len() && !b
            invariant 
                0 <= i <= v@.len(),
                b <==> exists|k: int| 0 <= k < i && v@[k] < 0,
                b ==> i > 0 && v@[(i-1) as int] < 0 && positive(v@.subrange(0, i-1))
            decreases v@.len() - i
        {
            b = v[i] < 0;
            i = i + 1;
        }
        
        if b {
            i = i - 1;
        }
        
        (b, i)
    }

    // Second method: alternative implementation
    fn mfirstNegative2(v: &[i32]) -> (result: (bool, usize))
        ensures 
            result.0 <==> exists|k: int| 0 <= k < v@.len() && v@[k] < 0,
            result.0 ==> {
                let i = result.1 as int;
                0 <= i < v@.len() && v@[i] < 0 && positive(v@.subrange(0, i))
            }
    {
        let mut i: usize = 0;
        let mut b: bool = false;
        
        while i < v.len() && !b
            invariant 
                0 <= i <= v@.len(),
                b ==> i < v@.len() && v@[i as int] < 0 && !(exists|k: int| 0 <= k < i && v@[k] < 0),
                b <== exists|k: int| 0 <= k < i && v@[k] < 0
            decreases v@.len() - i - (if b { 1int } else { 0int })
        {
            b = v[i] < 0;
            if !b {
                i = i + 1;
            }
        }
        
        (b, i)
    }
}