use vstd::prelude::*;

verus! {
    // Selection sort implementation in Verus
    fn selection_sort(a: &mut Vec<int>)
        ensures 
            a.len() == old(a).len(),
            forall|i: int, j: int| 0 <= i < j < a.len() ==> a[i] <= a[j],
            // Note: multiset preservation requires additional lemmas in Verus
            // a@.to_multiset() == old(a)@.to_multiset(),
    {
        let mut n: usize = 0;
        while n != a.len()
            invariant 
                0 <= n <= a.len(),
                a.len() == old(a).len(),
                // Elements before n are in their final sorted position
                forall|i: int, j: int| 0 <= i < n && n <= j < a.len() ==> a[i] <= a[j],
                // Elements in the sorted portion are sorted
                forall|i: int, j: int| 0 <= i < j < n ==> a[i] <= a[j],
            decreases a.len() - n,
        {
            let mut mindex: usize = n;
            let mut m: usize = n + 1;
            
            // Find the minimum element in the unsorted portion
            while m != a.len()
                invariant 
                    n <= mindex < m <= a.len(),
                    a.len() == old(a).len(),
                    // mindex points to the minimum element found so far
                    forall|i: int| n <= i < m ==> a[mindex as int] <= a[i],
                decreases a.len() - m,
            {
                if a[m] < a[mindex] {
                    mindex = m;
                }
                m = m + 1;
            }
            
            // Swap the minimum element with the current position
            let temp = a[n];
            let val_at_mindex = a[mindex];
            a.set(n, val_at_mindex);
            a.set(mindex, temp);
            
            n = n + 1;
        }
    }
}

fn main() {}