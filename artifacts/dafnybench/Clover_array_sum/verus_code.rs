use vstd::prelude::*;

verus! {
    fn arraySum(a: &[int], b: &[int]) -> (c: Vec<int>)
        requires 
            a.len() == b.len(),
        ensures 
            c.len() == a.len(),
            forall|i: int| #![auto] 0 <= i < a.len() ==> a[i] + b[i] == c[i],
    {
        let mut c = Vec::new();
        let mut i: usize = 0;
        
        while i < a.len()
            invariant
                0 <= i <= a.len(),
                c.len() == i,
                forall|j: int| #![auto] 0 <= j < i ==> a[j] + b[j] == c[j],
                a.len() == b.len(),
            decreases a.len() - i,
        {
            c.push(a[i] + b[i]);
            i = i + 1;
        }
        
        c
    }
}