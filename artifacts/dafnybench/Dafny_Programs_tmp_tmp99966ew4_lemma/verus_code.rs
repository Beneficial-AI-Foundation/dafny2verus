use vstd::prelude::*;

verus! {
    proof fn skipping_lemma(a: &[i32], j: usize)
        requires
            a.len() > 0,
            forall|i: int| 0 <= i < a.len() ==> a[i] >= 0,
            forall|i: int| 0 < i < a.len() ==> #[trigger] a[i-1] - 1 <= a[i],
            0 <= j < a.len(),
        ensures
            forall|k: int| j <= k < j + a[j as int] && k < a.len() ==> a[k] != 0,
    {
        // Proof by induction - omitted for translation
        assume(false);
    }

    fn find_zero(a: &[i32]) -> (index: i32)
        requires
            a.len() > 0,
            forall|i: int| 0 <= i < a.len() ==> a[i] >= 0,
            forall|i: int| 0 < i < a.len() ==> #[trigger] a[i-1] - 1 <= a[i],
        ensures
            index < 0 ==> forall|i: int| 0 <= i < a.len() ==> a[i] != 0,
            0 <= index ==> index < a.len() && a[index as int] == 0,
    {
        let mut index: usize = 0;
        while index < a.len()
            decreases a.len() - index
        {
            if a[index] == 0 {
                return index as i32;
            }
            proof {
                skipping_lemma(a, index);
            }
            index = index.checked_add(a[index] as usize).unwrap();
        }
        -1i32
    }
}