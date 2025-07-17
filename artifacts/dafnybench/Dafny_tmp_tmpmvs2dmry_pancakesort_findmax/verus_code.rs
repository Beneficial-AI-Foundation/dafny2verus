use vstd::prelude::*;

verus! {
    // returns an index of the largest element of array 'a' in the range [0..n)
    fn findMax(a: &[int], n: usize) -> (r: usize)
        requires 
            a.len() > 0,
            0 < n <= a.len(),
        ensures 
            0 <= r < n <= a.len(),
            forall|k: int| 0 <= k < n ==> a[r as int] >= a[k],
            // Array is unchanged (implicit in Rust with & reference)
    {
        let mut mi: usize = 0;
        let mut i: usize = 0;
        
        while i < n
            invariant 
                0 <= i <= n <= a.len(),
                0 <= mi < n,
                forall|k: int| 0 <= k < i ==> a[mi as int] >= a[k],
            decreases n - i,
        {
            if a[i] > a[mi] {
                mi = i;
            }
            i = i + 1;
        }
        mi
    }
}