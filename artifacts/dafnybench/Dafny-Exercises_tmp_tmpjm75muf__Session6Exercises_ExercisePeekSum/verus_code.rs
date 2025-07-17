use vstd::prelude::*;

verus! {
    // Predicate to check if element at index i is a "peek" 
    // (greater than or equal to all previous elements)
    spec fn isPeek(v: &[int], i: int) -> bool {
        &&& 0 <= i < v.len()
        &&& forall|k: int| 0 <= k < i ==> v[i] >= v[k]
    }

    // Recursive function to sum all "peek" elements up to index i
    spec fn peekSum(v: &[int], i: int) -> int 
        decreases i
    {
        if i == 0 {
            0
        } else if 0 <= i <= v.len() && isPeek(v, i - 1) {
            v[i - 1] + peekSum(v, i - 1)
        } else if 0 <= i <= v.len() {
            peekSum(v, i - 1)
        } else {
            0
        }
    }

    // O(n) algorithm to compute the sum of all peek elements
    fn mPeekSum(v: &[int]) -> (sum: int)
        requires v.len() > 0
        ensures sum == peekSum(v, v.len() as int)
    {
        let mut i: usize = 1;
        let mut sum: int = v[0];
        let mut lmax: int = v[0];
        
        // Establish initial invariant: sum == peekSum(v, 1)
        // Since isPeek(v, 0) is vacuously true (no elements before index 0)
        assert(isPeek(v, 0));
        assert(peekSum(v, 1) == v[0] + peekSum(v, 0));
        assert(peekSum(v, 0) == 0);
        assert(sum == peekSum(v, 1));
        
        while i < v.len()
            invariant 
                0 < i <= v.len(),
                // lmax is the maximum element seen so far
                exists|k: int| 0 <= k < i && v[k] == lmax,
                forall|k: int| 0 <= k < i ==> lmax >= v[k],
                // sum equals the peek sum up to current index
                sum == peekSum(v, i as int)
            decreases v.len() - i
        {
            if v[i] >= lmax {
                // v[i] is a new peek element
                sum = sum + v[i];
                lmax = v[i];
            }
            i = i + 1;
        }
        
        sum
    }
}