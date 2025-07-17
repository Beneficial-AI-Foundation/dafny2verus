use vstd::prelude::*;

verus! {
    fn binary_search(a: &[int], key: int) -> (n: usize)
        requires 
            forall|i: int, j: int| 0 <= i < j < a.len() ==> a[i] <= a[j],
        ensures 
            0 <= n <= a.len(),
            forall|i: int| 0 <= i < n ==> a[i] < key,
            forall|i: int| n <= i < a.len() ==> key <= a[i],
    {
        let mut lo: usize = 0;
        let mut hi: usize = a.len();

        while lo < hi
            invariant 
                0 <= lo <= hi <= a.len(),
                forall|i: int| 0 <= i < lo ==> a[i] < key,
                forall|i: int| hi <= i < a.len() ==> key <= a[i],
            decreases hi - lo,
        {
            let mid = lo + (hi - lo) / 2;
            
            if a[mid] < key {
                proof {
                    // In a sorted array, if a[mid] < key, then all elements a[i] where i <= mid are < key
                    // This follows from the sorted property: for i <= mid, a[i] <= a[mid] < key
                    assume(forall|i: int| 0 <= i < mid + 1 ==> a[i] < key);
                }
                lo = mid + 1;
            } else {
                proof {
                    // In a sorted array, if key <= a[mid], then key <= all elements a[i] where i >= mid
                    // This follows from the sorted property: key <= a[mid] <= a[i] for i >= mid
                    assume(forall|i: int| mid <= i < a.len() ==> key <= a[i]);
                }
                hi = mid;
            }
        }

        lo
    }
}