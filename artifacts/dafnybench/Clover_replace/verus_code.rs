use vstd::prelude::*;

verus! {
    fn replace(arr: &mut Vec<i32>, k: i32)
        ensures
            forall|i: int| #![auto] 0 <= i < old(arr)@.len() ==> old(arr)@.index(i) > k ==> arr@.index(i) == -1,
            forall|i: int| #![auto] 0 <= i < old(arr)@.len() ==> old(arr)@.index(i) <= k ==> arr@.index(i) == old(arr)@.index(i),
    {
        let mut i: usize = 0;
        let ghost old_arr = arr@;
        
        while i < arr.len() {
            decreases(arr.len() - i);
            invariant(
                0 <= i <= arr.len() &&
                arr@.len() == old_arr.len() &&
                forall|j: int| #![auto] 0 <= j < i ==> old_arr.index(j) > k ==> arr@.index(j) == -1 &&
                forall|j: int| #![auto] 0 <= j < i ==> old_arr.index(j) <= k ==> arr@.index(j) == old_arr.index(j) &&
                forall|j: int| #![auto] i <= j < arr.len() ==> old_arr.index(j) == arr@.index(j)
            );
            
            if arr[i] > k {
                arr.set(i, -1);
            }
            
            i = i + 1;
        }
    }
}