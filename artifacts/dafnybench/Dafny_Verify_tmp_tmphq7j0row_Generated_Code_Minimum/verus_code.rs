use vstd::prelude::*;

verus! {
    fn minimum(a: &[int]) -> (m: int)
        requires a.len() > 0,
        ensures 
            exists|i: int| 0 <= i < a.len() && m == a[i],
            forall|i: int| 0 <= i < a.len() ==> m <= a[i]
    {
        let mut n: usize = 0;
        let mut m: int = a[0];
        
        while n != a.len()
            invariant 
                0 <= n <= a.len(),
                exists|i: int| 0 <= i < a.len() && m == a[i],
                forall|i: int| 0 <= i < n ==> m <= a[i]
            decreases a.len() - n
        {
            if a[n] < m {
                m = a[n];
            }
            n = n + 1;
        }
        
        m
    }
}