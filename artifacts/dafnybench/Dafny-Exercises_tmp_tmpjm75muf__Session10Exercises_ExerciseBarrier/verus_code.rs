use vstd::prelude::*;

verus! {
    fn barrier(v: &[int], p: usize) -> (b: bool)
        requires 
            v.len() > 0,
            p < v.len(),
        ensures 
            b == (forall|k: int, l: int| 0 <= k <= p && p < l < v.len() ==> v[k] < v[l])
    {
        let mut i: usize = 1;
        let mut max: usize = 0;
        
        while i <= p
            invariant 
                i <= p + 1,
                max < i,
                max < v.len(),
                forall|k: int| 0 <= k < i && k < v.len() ==> v[max as int] >= v[k],
            decreases p + 1 - i
        {
            if v[i] > v[max] {
                max = i;
            }
            
            i = i + 1;
        }

        while i < v.len() && v[i] > v[max]
            invariant 
                p < i <= v.len(),
                max <= p,
                max < v.len(),
                forall|k: int| 0 <= k <= p && k < v.len() ==> v[k] <= v[max as int],
                forall|k: int| p < k < i && k < v.len() ==> v[k] > v[max as int],
            decreases v.len() - i
        {
            i = i + 1;
        }
        
        i == v.len()
    }
}

fn main() {}