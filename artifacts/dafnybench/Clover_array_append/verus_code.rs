use vstd::prelude::*;

verus! {
    fn append(a: &[int], b: int) -> (c: Vec<int>)
        requires a.len() < usize::MAX
        ensures c@ == a@ + seq![b]
    {
        let mut c = Vec::with_capacity(a.len() + 1);
        let mut i: usize = 0;
        
        while i < a.len()
            invariant 
                0 <= i <= a.len(),
                c.len() == i,
                forall|ii: int| 0 <= ii < i ==> c[ii] == a[ii]
            decreases a.len() - i
        {
            c.push(a[i]);
            i = i + 1;
        }
        c.push(b);
        c
    }
}

fn main() {}