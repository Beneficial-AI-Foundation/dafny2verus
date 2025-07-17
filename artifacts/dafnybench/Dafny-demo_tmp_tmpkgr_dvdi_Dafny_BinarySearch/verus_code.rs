use vstd::prelude::*;

verus! {
    spec fn sorted(a: &[int], l: int, u: int) -> bool {
        forall|i: int, j: int| 0 <= l <= i <= j <= u < a.len() ==> a[i] <= a[j]
    }

    fn binary_search(a: &[int], key: int) -> (index: i32)
        requires 
            a.len() > 0,
            a.len() <= 2147483647,
            sorted(a, 0, (a.len() - 1) as int),
        ensures 
            index >= 0 ==> index < a.len() && a[index as int] == key,
            index < 0 ==> forall|k: int| 0 <= k < a.len() ==> a[k] != key,
    {
        let mut low: usize = 0;
        let mut high: usize = a.len();
        
        while low < high
            invariant 
                0 <= low <= high <= a.len(),
                forall|i: int| 0 <= i < a.len() && !(low <= i < high) ==> a[i] != key,
            decreases high - low,
        {
            let mid: usize = low + (high - low) / 2;
            
            assert(low <= mid < high);
            
            let mid_val = a[mid];
            
            if mid_val < key {
                low = mid + 1;
            } else if key < mid_val {
                high = mid;
            } else {
                assert(mid_val == key);
                assert(mid < a.len());
                assert(a[mid as int] == key);
                return mid as i32;
            }
        }
        
        assert(low == high);
        -1
    }
}