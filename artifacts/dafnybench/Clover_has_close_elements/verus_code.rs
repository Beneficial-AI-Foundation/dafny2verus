use vstd::prelude::*;

verus! {
    spec fn abs_diff(a: int, b: int) -> int {
        if a >= b { a - b } else { b - a }
    }

    fn has_close_elements(numbers: Seq<int>, threshold: int) -> (res: bool)
        requires threshold >= 0,
        ensures ({
            let close_exists = exists|i: int, j: int| 0 <= i < numbers.len() && 0 <= j < numbers.len() && i != j && 
                abs_diff(numbers[i], numbers[j]) < threshold;
            let all_far = forall|i: int, j: int| 1 <= i < numbers.len() && 0 <= j < i ==> 
                abs_diff(numbers[i], numbers[j]) >= threshold;
            (res ==> close_exists) && (!res ==> all_far)
        })
    {
        let mut res = false;
        let mut idx: usize = 0;
        
        while idx < numbers.len() // Type conversion issue here
            invariant 
                0 <= idx <= numbers.len(),
                !res,
                forall|i: int, j: int| 0 <= i < idx && 0 <= j < i ==> 
                    abs_diff(numbers[i], numbers[j]) >= threshold
        {
            let mut idx2: usize = 0;
            
            while idx2 < idx
                invariant 
                    0 <= idx <= numbers.len(),
                    0 <= idx2 <= idx,
                    !res,
                    forall|j: int| 0 <= j < idx2 ==> 
                        abs_diff(numbers[idx as int], numbers[j]) >= threshold
            {
                let ghost num_idx = numbers[idx as int];
                let ghost num_idx2 = numbers[idx2 as int];
                let ghost distance = abs_diff(num_idx2, num_idx);
                
                if distance < threshold {
                    res = true;
                    return res;
                }
                
                idx2 = idx2 + 1;
            }
            
            idx = idx + 1;
        }
        
        res
    }
}