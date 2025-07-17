use vstd::prelude::*;

verus! {
    fn reverse(a: &mut Vec<i32>)
        requires old(a).len() > 0,
        ensures 
            a.len() == old(a).len(),
            forall|i: int| 0 <= i < a.len() ==> a[i] == old(a)[a.len() - 1 - i],
    {
        let mut i: usize = 0;
        while i < a.len() / 2
            invariant 
                0 <= i <= a.len() / 2,
                a.len() == old(a).len(),
                forall|k: int| (0 <= k < i || (a.len() as int - 1 - i as int) < k <= a.len() - 1) ==> 
                    a[k] == old(a)[a.len() - 1 - k],
                forall|k: int| i <= k <= (a.len() as int - 1 - i as int) ==> 
                    a[k] == old(a)[k],
            decreases a.len() - i,
        {
            let len = a.len();
            let left_val = a[i];
            let right_val = a[len - 1 - i];
            a.set(i, right_val);
            a.set(len - 1 - i, left_val);
            i = i + 1;
        }
    }
}