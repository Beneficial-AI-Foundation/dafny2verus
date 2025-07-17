use vstd::prelude::*;

verus! {
    // Works by dividing the input list into two parts: sorted and unsorted. At the beginning, 
    // the sorted part is empty and the unsorted part contains all the elements.
    fn selection_sort(a: &mut Vec<i32>)
        requires old(a).len() > 0,
        ensures 
            // Ensures the final array is sorted in ascending order
            forall|i: int, j: int| 0 <= i < j < a.len() ==> a[i] <= a[j],
            // Ensures that the final array has the same length as the initial array
            a.len() == old(a).len(),
    {
        let mut n: usize = 0;
        while n != a.len()
            invariant 
                // Ensures that n is always within the bounds of the array
                0 <= n <= a.len(),
                // Guarantees that the portion of the array up to index n is sorted
                forall|i: int, j: int| 0 <= i < j < a.len() && j < n ==> a[i] <= a[j],
                // Guarantees that all elements before n are less than or equal to elements after and at n
                forall|i: int, j: int| #![trigger a[i], a[j]] 0 <= i < n && n <= j < a.len() ==> a[i] <= a[j],
                // Ensures that the array has the same length
                a.len() == old(a).len(),
            decreases a.len() - n,
        {
            let mut mindex: usize = n;
            let mut m: usize = n;
            while m != a.len()
                invariant 
                    // Ensures that mindex is always within the bounds of the array
                    n <= mindex < a.len(),
                    n <= m <= a.len(),
                    // Ensures that a[mindex] is the smallest element from a[n] to a[m-1]
                    forall|i: int| #![trigger a[i]] n <= i < m ==> a[mindex as int] <= a[i],
                    // Ensures that the array has the same length
                    a.len() == old(a).len(),
                decreases a.len() - m,
            {
                if a[m] < a[mindex] {
                    mindex = m;
                }
                m = m + 1;
            }
            // Swaps the first element of the unsorted array with the current smallest element
            // in the unsorted part if it is smaller
            if a[mindex] < a[n] {
                let temp_mindex = a[mindex];
                let temp_n = a[n];
                a.set(mindex, temp_n);
                a.set(n, temp_mindex);
            }
            n = n + 1;
        }
    }
}