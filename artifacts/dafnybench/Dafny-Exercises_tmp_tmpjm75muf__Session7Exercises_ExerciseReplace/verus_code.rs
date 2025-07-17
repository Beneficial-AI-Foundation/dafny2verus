use vstd::prelude::*;

verus! {
    fn replace(v: &mut Vec<int>, x: int, y: int)
        requires old(v).len() >= 0,
        ensures 
            v.len() == old(v).len(),
            forall|k: int| 0 <= k < old(v).len() && old(v)[k] == x ==> v[k] == y,
            forall|k: int| 0 <= k < old(v).len() && old(v)[k] != x ==> v[k] == old(v)[k],
    {
        let mut i: usize = 0;
        while i < v.len()
            invariant 
                0 <= i <= v.len(),
                v.len() == old(v).len(),
                forall|k: int| 0 <= k < i && old(v)[k] == x ==> v[k] == y,
                forall|k: int| i <= k < v.len() ==> v[k] == old(v)[k],
                forall|k: int| 0 <= k < i && old(v)[k] != x ==> v[k] == old(v)[k],
            decreases v.len() - i,
        {
            if v[i] == x {
                v.set(i, y);
            }
            i = i + 1;
        }
    }
}