use vstd::prelude::*;

verus! {
    // Recursive function to compute sum of array elements from index 0 to i
    spec fn sum(a: &[int], i: int) -> int
        recommends 0 <= i < a.len()
        decreases i
    {
        if i < 0 || i >= a.len() {
            0  // Default value for out-of-bounds
        } else {
            a[i] + if i == 0 { 0 } else { sum(a, i - 1) }
        }
    }

    // Method to compute cumulative sum array
    fn cumsum(a: &[int], b: &mut [int])
        requires 
            a.len() == old(b).len(),  // Arrays must have same length
            a.len() > 0,              // Arrays must be non-empty
        ensures 
            b.len() == a.len(),
            forall|i: int| 0 <= i < a.len() ==> b@[i] == sum(a, i)
    {
        // Initialize first element
        b[0] = a[0];
        
        let mut i: usize = 1;
        
        // Compute cumulative sum for remaining elements
        while i < a.len()
            invariant 
                1 <= i <= a.len(),
                b.len() == a.len(),
                forall|i_: int| 0 <= i_ < i ==> b@[i_] == sum(a, i_)
            decreases a.len() - i
        {
            let ghost prev_sum = b@[i as int - 1];
            
            // Cumulative sum: current element + previous cumulative sum
            b[i] = b[i - 1] + a[i];
            
            // Proof that b[i] == sum(a, i)
            proof {
                assert(prev_sum == sum(a, i as int - 1));
                assert(sum(a, i as int) == a[i as int] + sum(a, i as int - 1));
                assert(b@[i as int] == prev_sum + a[i as int]);
                assert(b@[i as int] == sum(a, i as int - 1) + a[i as int]);
                assert(b@[i as int] == sum(a, i as int));
            }
            
            i = i + 1;
        }
    }
}