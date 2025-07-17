use vstd::prelude::*;

verus! {
    fn selection_sort(a: &mut Vec<i32>)
        ensures 
            a.len() == old(a).len(),
            forall|i: int, j: int| 0 <= i < j < a.len() ==> a[i] <= a[j],
    {
        let mut n: usize = 0;
        while n != a.len()
            invariant 
                0 <= n <= a.len(),
                a.len() == old(a).len(),
                // Sorted portion invariant
                forall|i: int, j: int| 0 <= i < j < a.len() && j < n ==> a[i] <= a[j],
                // Partition invariant: sorted elements <= unsorted elements
                forall|i: int, j: int| 0 <= i < n && n <= j < a.len() ==> a[i] <= a[j],
            decreases a.len() - n,
        {
            let mut mindex: usize = n;
            let mut m: usize = n;
            while m != a.len()
                invariant 
                    n <= mindex < a.len(),
                    n <= m <= a.len(),
                    a.len() == old(a).len(),
                    // mindex contains the minimum element found so far
                    forall|i: int| n <= i < m ==> a[mindex as int] <= a[i],
                decreases a.len() - m,
            {
                if a[m] < a[mindex] {
                    mindex = m;
                }
                m = m + 1;
            }
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