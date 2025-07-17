use vstd::prelude::*;

verus! {
    fn twoSum(nums: &[int], target: int) -> (result: (usize, usize))
        requires 
            nums.len() > 1,
            exists|i: int, j: int| 0 <= i < j < nums.len() && nums@[i] + nums@[j] == target,
        ensures 
            ({
                let (i, j) = result;
                0 <= i < j < nums.len() && nums@[i as int] + nums@[j as int] == target
            }),
    {
        let n = nums.len();
        let mut i = 0usize;
        
        while i < n - 1
            invariant 
                0 <= i <= n - 1,
                forall|ii: int, jj: int| (0 <= ii < i && ii < jj < n) ==> nums@[ii] + nums@[jj] != target,
            decreases n - i,
        {
            let mut j = i + 1;
            while j < n
                invariant 
                    0 <= i < n - 1,
                    i + 1 <= j <= n,
                    forall|jj: int| (i as int) < jj < (j as int) ==> nums@[i as int] + nums@[jj] != target,
                decreases n - j,
            {
                if nums[i] + nums[j] == target {
                    return (i, j);
                }
                j = j + 1;
            }
            i = i + 1;
        }
        
        proof { 
            // We should never reach here because of the precondition
            assert(false);
        }
        (0, 0)
    }
}