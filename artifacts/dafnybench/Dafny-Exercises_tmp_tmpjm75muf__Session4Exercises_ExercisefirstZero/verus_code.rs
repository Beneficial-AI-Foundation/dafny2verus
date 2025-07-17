use vstd::prelude::*;

verus! {
    fn mfirstCero(v: &[i32]) -> (i: usize)
        ensures 
            0 <= i <= v.len(),
            forall|j: int| 0 <= j < i ==> v@[j] != 0,
            i != v.len() ==> v@[i as int] == 0
    {
        let mut i: usize = 0;
        while i < v.len() && v[i] != 0
            invariant 
                0 <= i <= v.len(),
                forall|j: int| 0 <= j < i ==> v@[j] != 0
            decreases v.len() - i
        {
            i = i + 1;
        }
        i
    }
}