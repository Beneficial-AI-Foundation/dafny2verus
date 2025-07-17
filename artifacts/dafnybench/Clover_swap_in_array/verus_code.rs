use vstd::prelude::*;

verus! {
    fn swap(arr: &mut Vec<i32>, i: usize, j: usize)
        requires 
            i < old(arr).len(),
            j < old(arr).len(),
        ensures
            arr[i as int] == old(arr)[j as int],
            arr[j as int] == old(arr)[i as int],
            forall|k: int| 0 <= k < old(arr).len() && k != i && k != j ==> arr[k] == old(arr)[k],
    {
        let tmp = arr[i];
        let val_j = arr[j];
        arr.set(i, val_j);
        arr.set(j, tmp);
    }
}