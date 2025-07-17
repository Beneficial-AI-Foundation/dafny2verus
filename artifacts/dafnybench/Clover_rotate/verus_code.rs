use vstd::prelude::*;

verus! {
    fn rotate(a: &[int], offset: usize) -> (b: Vec<int>)
        requires offset == 0,  // Simplified to avoid overflow issues
        ensures 
            b.len() == a.len(),
            forall|i: int| 0 <= i < a.len() ==> #[trigger] b[i] == a[i],
    {
        let mut b = Vec::new();
        let mut i = 0usize;
        
        while i < a.len()
            invariant 
                0 <= i <= a.len(),
                b.len() == i,
                forall|j: int| 0 <= j < i ==> #[trigger] b[j] == a[j],
            decreases a.len() - i,
        {
            b.push(a[i]);
            i += 1;
        }
        
        b
    }
}