use vstd::prelude::*;

verus! {
    fn add_small_numbers(a: &[i32], n: usize, max: i32) -> (r: i32)
        requires
            n > 0,
            n <= a.len(),
            forall|i: int| 0 <= i && i < n ==> a[i] <= max,
            max >= 0,
            n <= 100,
            max <= 100,
        ensures
            r <= max * (n as i32),
    {
        let mut i: usize = 0;
        let mut r: i32 = 0;

        while i < n
            invariant
                0 <= i <= n,
                r <= max * (i as i32),
                n <= a.len(),
                forall|j: int| 0 <= j && j < n ==> a[j] <= max,
                max >= 0,
                n <= 100,
                max <= 100,
            decreases n - i
        {
            r = r + a[i];
            i = i + 1;
        }
        r
    }
}