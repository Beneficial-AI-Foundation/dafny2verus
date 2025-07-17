use vstd::prelude::*;

verus! {
    fn find_max(a: &[int]) -> (i: usize)
        requires a.len() > 0,
        ensures 
            i < a.len(),
            forall|k: int| 0 <= k < a.len() ==> a[k] <= a[i as int],
    {
        let mut i: usize = 0;
        let mut index: usize = 1;
        
        while index < a.len()
            invariant 
                0 < index <= a.len(),
                i < index,
                forall|k: int| 0 <= k < index ==> a[k] <= a[i as int],
            decreases a.len() - index,
        {
            if a[index] > a[i] {
                i = index;
            }
            index = index + 1;
        }
        
        i
    }
}