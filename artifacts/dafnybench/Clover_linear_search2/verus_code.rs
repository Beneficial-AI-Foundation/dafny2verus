use vstd::prelude::*;

verus! {
    fn linear_search(a: &[int], e: int) -> (n: usize)
        requires 
            exists|i: int| 0 <= i < a.len() && a[i] == e,
        ensures 
            0 <= n < a.len(),
            a[n as int] == e,
            forall|k: int| 0 <= k < n ==> a[k] != e,
    {
        let mut n: usize = 0;
        while n != a.len()
            invariant 
                0 <= n <= a.len(),
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

fn main() {}