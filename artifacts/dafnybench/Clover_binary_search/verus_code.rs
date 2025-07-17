use vstd::prelude::*;

verus! {
    #[verifier::loop_isolation(false)]
    fn binary_search(a: &[int], key: int) -> (n: usize)
        requires 
            forall|i: int, j: int| 0 <= i < j < a.len() ==> a[i] <= a[j],
        ensures 
            0 <= n <= a.len(),
            forall|i: int| 0 <= i < n ==> a[i] < key,
            n == a.len() ==> forall|i: int| 0 <= i < a.len() ==> a[i] < key,
            forall|i: int| n <= i < a.len() ==> a[i] >= key,
    {
        let mut lo: usize = 0;
        let mut hi: usize = a.len();
        
        while lo < hi
            invariant 
                0 <= lo <= hi <= a.len(),
                forall|i: int| 0 <= i < lo ==> a[i] < key,
                forall|i: int| hi <= i < a.len() ==> a[i] >= key,
            decreases hi - lo,
        {
            let mid = lo + (hi - lo) / 2;
            assert(lo <= mid < hi);
            
            if a[mid] < key {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo
    }
}

fn main() {}