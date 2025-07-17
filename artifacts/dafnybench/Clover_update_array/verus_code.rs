use vstd::prelude::*;

verus! {
    fn update_elements(a: &mut Vec<i32>)
        requires 
            old(a).len() >= 8,
            old(a)[4] + 3 <= i32::MAX,
        ensures
            old(a)[4] + 3 == a[4],
            a[7] == 516,
            forall|i: int| 0 <= i < a.len() && i != 7 && i != 4 ==> a[i] == old(a)[i],
    {
        let old_val = a[4];
        a.set(4, old_val + 3);
        a.set(7, 516);
    }

    fn main() {}
}