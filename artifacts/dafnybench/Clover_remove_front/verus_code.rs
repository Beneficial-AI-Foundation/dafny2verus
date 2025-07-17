use vstd::prelude::*;

verus! {
    fn remove_front(a: &[int]) -> (c: Vec<int>)
        requires a.len() > 0,
        ensures c@ == a@.subrange(1, a.len() as int),
    {
        let mut c = Vec::new();
        let mut i: usize = 1;
        
        while i < a.len()
            invariant 
                1 <= i <= a.len(),
                c.len() == i - 1,
                forall|ii: int| 1 <= ii < i ==> c[ii - 1] == a[ii],
            decreases a.len() - i,
        {
            c.push(a[i]);
            i = i + 1;
        }
        
        c
    }
}