use vstd::prelude::*;

verus! {
    spec fn contains(v: int, a: &[int], n: int) -> bool {
        exists|j: int| 0 <= j < n && a[j] == v
    }

    spec fn upper_bound(v: int, a: &[int], n: int) -> bool {
        forall|j: int| 0 <= j < n ==> a[j] <= v
    }

    spec fn is_max(m: int, a: &[int], n: int) -> bool {
        contains(m, a, n) && upper_bound(m, a, n)
    }

    fn max(a: &[int], n: usize) -> (max: int)
        requires 
            0 < n <= a.len(),
        ensures 
            is_max(max, a, n as int),
    {
        let mut i: usize = 1;
        let mut max: int = a[0];

        while i < n
            invariant
                i <= n,
                n <= a.len(),
                is_max(max, a, i as int),
            decreases n - i,
        {
            if a[i] > max {
                max = a[i];
            }
            i = i + 1;
        }
        max
    }
}