use vstd::prelude::*;

verus! {
    fn addArrays(a: &[i32], b: &[i32]) -> (c: Vec<i32>)
        requires 
            a.len() == b.len(),
        ensures 
            b.len() == c.len(),
            forall|i: int| 0 <= i < c.len() ==> #[trigger] c[i] == a[i] + b[i],
    {
        let mut c = Vec::new();
        let mut j: usize = 0;
        
        while j < a.len()
            invariant 
                c.len() == j,
                j <= a.len(),
                a.len() == b.len(),
                forall|i: int| 0 <= i < j ==> #[trigger] c[i] == a[i] + b[i],
            decreases a.len() - j,
        {
            assume(a[j as int] + b[j as int] >= i32::MIN && a[j as int] + b[j as int] <= i32::MAX);
            c.push(a[j] + b[j]);
            j = j + 1;
        }
        
        c
    }
}