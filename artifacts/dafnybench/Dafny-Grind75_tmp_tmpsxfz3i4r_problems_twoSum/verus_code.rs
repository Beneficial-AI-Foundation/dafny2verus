use vstd::prelude::*;

verus! {
    // Predicate to check if two indices form a summing pair
    spec fn summingPair(i: int, j: int, nums: Vec<int>, target: int) -> bool
        recommends 0 <= i < nums.len() && 0 <= j < nums.len()
    {
        i != j && nums[i] + nums[j] == target
    }

    fn twoSum(nums: Vec<int>, target: int) -> (pair: (usize, usize))
        requires 
            exists|i: int, j: int| 0 <= i < j < nums.len() && summingPair(i, j, nums, target) &&
            forall|l: int, m: int| 0 <= l < m < nums.len() && l != i && m != j ==> !summingPair(l, m, nums, target)
        ensures 
            0 <= pair.0 < nums.len() && 0 <= pair.1 < nums.len() && 
            summingPair(pair.0 as int, pair.1 as int, nums, target)
    {
        let mut pair = (0usize, 0usize);
        let mut i: usize = 0;
        let len = nums.len();
        
        while i < len
            invariant 
                i <= len,
                len == nums.len(),
                forall|z: int, j: int| 0 <= z < i && z + 1 <= j < nums.len() ==> !summingPair(z, j, nums, target)
            decreases len - i
        {
            let mut k: usize = i + 1;
            while k < len
                invariant 
                    i + 1 <= k <= len,
                    len == nums.len(),
                    forall|q: int| i + 1 <= q < k ==> !summingPair(i as int, q, nums, target)
                decreases len - k
            {
                if nums[i] + nums[k] == target {
                    pair = (i, k);
                    return pair;
                }
                k = k + 1;
            }
            i = i + 1;
        }
        
        pair
    }
}