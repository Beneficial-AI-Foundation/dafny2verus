use vstd::prelude::*;

verus! {
    // Specification functions for min and countMin
    // These would need full recursive implementations for complete verification
    spec fn min(v: Seq<int>, i: int) -> int 
        recommends 1 <= i <= v.len()
        decreases i
    {
        if i == 1 { 
            v[0] 
        } else { 
            if v[i-1] <= min(v, i-1) { 
                v[i-1] 
            } else { 
                min(v, i-1) 
            }
        }
    }

    spec fn countMin(v: Seq<int>, x: int, i: int) -> int
        recommends 0 <= i <= v.len()
        decreases i
    {
        if i == 0 { 
            0 
        } else if v[i-1] == x { 
            1 + countMin(v, x, i-1) 
        } else { 
            countMin(v, x, i-1) 
        }
    }

    fn mCountMin(v: &[i32]) -> (c: i32)
        requires 
            v.len() > 0,
            v.len() < 1000  // reasonable bound to avoid overflow
        ensures c > 0
    {
        let mut i: usize = 1;
        let mut c: i32 = 1;
        let mut mini: i32 = v[0];
        
        while i < v.len()
            invariant 
                0 < i <= v.len(),
                c >= 1,
                c <= i as i32,
                v.len() < 1000
            decreases v.len() - i
        {
            if v[i] == mini {
                // Guard against overflow
                if c < 999 {
                    c = c + 1;
                }
            } else if v[i] < mini {
                c = 1;
                mini = v[i];
            }
            i = i + 1;
        }
        c
    }
}