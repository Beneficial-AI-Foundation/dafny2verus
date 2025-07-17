use vstd::prelude::*;

verus! {
    fn swap3(a: &mut Vec<int>, h: usize, i: usize, j: usize)
        requires 
            h < old(a).len(),
            i < old(a).len(),
            j < old(a).len(),
            i != j && j != h && h != i,
        ensures
            a@[h as int] == old(a)@[i as int],
            a@[j as int] == old(a)@[h as int],
            a@[i as int] == old(a)@[j as int],
            forall|k: int| 0 <= k < old(a).len() && k != h && k != i && k != j ==> a@[k] == old(a)@[k],
    {
        let tmp_h = a[h];
        let tmp_i = a[i];
        let tmp_j = a[j];
        a.set(h, tmp_i);
        a.set(i, tmp_j);
        a.set(j, tmp_h);
    }

    fn test_swap3(a: &mut Vec<int>, h: usize, i: usize, j: usize)
        requires 
            h < old(a).len(),
            i < old(a).len(),
            j < old(a).len(),
            i != j && j != h && h != i,
    {
        swap3(a, h, i, j);
        assert(a@[h as int] == old(a)@[i as int]);
        assert(a@[j as int] == old(a)@[h as int]);
        assert(a@[i as int] == old(a)@[j as int]);
        assert(forall|k: int| 0 <= k < old(a).len() && k != h && k != i && k != j ==> a@[k] == old(a)@[k]);
    }
}