use vstd::prelude::*;

verus! {
    fn linear_search(a: &[int], e: int) -> (n: usize)
        requires a.len() <= usize::MAX,
        ensures 
            n <= a.len(),
            n == a.len() || a[n as int] == e,
            forall|i: int| 0 <= i < n ==> e != a[i],
    {
        let mut n: usize = 0;
        while n != a.len()
            invariant 
                n <= a.len(),
                forall|i: int| 0 <= i < n ==> e != a[i],
            decreases a.len() - n,
        {
            if e == a[n] {
                return n;
            }
            n = n + 1;
        }
        n
    }
}