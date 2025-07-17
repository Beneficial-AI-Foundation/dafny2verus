use vstd::prelude::*;

verus! {
    fn iter_copy<T: Copy>(s: &[T]) -> (t: Vec<T>)
        ensures
            s.len() == t.len(),
            forall|i: int| 0 <= i < s.len() ==> s[i] == t[i]
    {
        let mut t = Vec::with_capacity(s.len());
        let mut i: usize = 0;
        
        while i < s.len()
            invariant
                0 <= i <= s.len(),
                t.len() == i,
                forall|x: int| 0 <= x < i ==> s[x] == t[x]
            decreases s.len() - i
        {
            t.push(s[i]);
            i = i + 1;
        }
        
        t
    }
}