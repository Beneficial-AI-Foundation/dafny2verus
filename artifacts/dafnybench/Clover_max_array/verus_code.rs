use vstd::prelude::*;

verus! {
    fn maxArray(a: &[int]) -> (m: int)
        requires a.len() >= 1,
        ensures
            forall|k: int| 0 <= k < a.len() ==> m >= a[k],
            exists|k: int| 0 <= k < a.len() && m == a[k]
    {
        let mut m = a[0];
        let mut index = 1;
        while index < a.len()
            invariant
                0 <= index <= a.len(),
                forall|k: int| 0 <= k < index ==> m >= a[k],
                exists|k: int| 0 <= k < index && m == a[k]
            decreases a.len() - index
        {
            m = if m > a[index as usize] { m } else { a[index as usize] };
            index = index + 1;
        }
        m
    }
}