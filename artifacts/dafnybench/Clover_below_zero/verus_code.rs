use vstd::prelude::*;

verus! {
    fn below_zero(operations: Vec<i32>) -> (result: (Vec<i32>, bool))
        requires operations.len() < usize::MAX
        ensures ({
            let (s, res) = result;
            &&& s.len() == operations.len() + 1
            &&& s[0] == 0
            &&& forall|i: int| 0 <= i < operations.len() ==> 
                #[trigger] s[i + 1] == s[i] + operations[i]
            &&& res == true ==> (exists|i: int| 1 <= i <= operations.len() && #[trigger] s[i] < 0)
            &&& res == false ==> forall|i: int| 0 <= i < s.len() ==> #[trigger] s[i] >= 0
        })
    {
        let mut result_bool = false;
        let mut s: Vec<i32> = Vec::new();
        s.push(0);
        
        let mut i: usize = 0;
        
        while i < operations.len()
            invariant 
                0 <= i <= operations.len(),
                s.len() == i + 1,
                s[0] == 0,
                forall|x: int| 0 <= x < i ==> 
                    #[trigger] s[x + 1] == s[x] + operations[x]
            decreases operations.len() - i
        {
            let current_val = s[i];
            let operation = operations[i];
            
            // Check for overflow
            assume(current_val.checked_add(operation).is_some());
            let next_val = current_val + operation;
            s.push(next_val);
            i = i + 1;
        }
        
        i = 0;
        while i < s.len()
            invariant 
                0 <= i <= s.len(),
                s.len() == operations.len() + 1,
                s[0] == 0,
                forall|x: int| 0 <= x < operations.len() ==> 
                    #[trigger] s[x + 1] == s[x] + operations[x],
                forall|x: int| 0 <= x < i ==> #[trigger] s[x] >= 0,
                result_bool == false
            decreases s.len() - i
        {
            if s[i] < 0 {
                result_bool = true;
                assert(exists|j: int| 1 <= j <= operations.len() && s[j] < 0) by {
                    if i == 0 {
                        // This shouldn't happen since s[0] = 0
                        assert(false);
                    } else {
                        assert(1 <= i <= operations.len());
                        assert(s[i as int] < 0);
                    }
                }
                return (s, result_bool);
            }
            i = i + 1;
        }
        
        // At this point, we've checked all elements and none are negative
        assert(forall|j: int| 0 <= j < s.len() ==> s[j] >= 0);
        (s, result_bool)
    }
}