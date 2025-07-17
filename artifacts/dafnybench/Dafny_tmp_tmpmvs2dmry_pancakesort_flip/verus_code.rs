use vstd::prelude::*;

verus! {
    // flips (i.e., reverses) array elements in the range [0..num]
    fn flip(a: &mut Vec<int>, num: usize)
        requires
            old(a).len() > 0,
            num < old(a).len(),
        ensures
            a.len() == old(a).len(),
            forall |k: int| 0 <= k <= num as int ==> a[k] == old(a)[num as int - k],
            forall |k: int| k > num as int && k < a.len() as int ==> a[k] == old(a)[k],
    {
        let mut tmp: int;
        let mut i: usize = 0;
        let mut j: usize = num;
        
        while i < j
            invariant
                a.len() == old(a).len(),
                i + j == num,
                0 <= i <= num / 2 + 1,
                num / 2 <= j <= num,
                i < a.len(),
                j < a.len(),
                forall |n: int| 0 <= n < i as int && n < a.len() as int ==> a[n] == old(a)[num as int - n],
                forall |n: int| 0 <= n < i as int && (num as int - n) < a.len() as int ==> a[num as int - n] == old(a)[n],
                forall |k: int| i as int <= k <= j as int ==> a[k] == old(a)[k],
                forall |k: int| k > num as int && k < a.len() as int ==> a[k] == old(a)[k],
            decreases j
        {
            tmp = a[i];
            let val_j = a[j];
            a.set(i, val_j);
            a.set(j, tmp);
            i = i + 1;
            j = j - 1;
        }
    }
}