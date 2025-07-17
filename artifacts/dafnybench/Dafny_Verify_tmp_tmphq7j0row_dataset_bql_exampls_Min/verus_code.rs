use vstd::prelude::*;

verus! {
    fn min(a: &[int], n: usize) -> (min: int)
        requires 
            0 < n <= a.len(),
        ensures
            exists|i: int| 0 <= i < n && a[i] == min,
            forall|i: int| 0 <= i < n ==> a[i] >= min,
    {
        let mut min = a[0];
        let mut i = 1;

        while i < n
            invariant 
                i <= n,
                n <= a.len(),
                exists|j: int| 0 <= j < i && a[j] == min,
                forall|j: int| 0 <= j < i ==> a[j] >= min,
            decreases n - i,
        {
            if a[i] < min {
                min = a[i];
            }
            i = i + 1;
        }
        min
    }
}

fn main() {}