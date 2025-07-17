use vstd::prelude::*;

verus! {
    fn rotate_right(a: &mut Vec<i32>)
        requires old(a).len() > 0,
        ensures 
            a.len() == old(a).len(),
            forall|i: int| 1 <= i < a.len() ==> a[i] == old(a)[i - 1],
            a[0] == old(a)[old(a).len() - 1]
    {
        let mut n: usize = 1;
        while n != a.len()
            invariant 
                a.len() == old(a).len(),
                1 <= n <= a.len(),
                forall|i: int| 1 <= i < n ==> a[i] == old(a)[i - 1],
                a[0] == old(a)[n - 1],
                forall|i: int| n <= i <= a.len() - 1 ==> a[i] == old(a)[i]
            decreases a.len() - n
        {
            let temp = a[0];
            let val_n = a[n];
            a.set(0, val_n);
            a.set(n, temp);
            n = n + 1;
        }
    }
}