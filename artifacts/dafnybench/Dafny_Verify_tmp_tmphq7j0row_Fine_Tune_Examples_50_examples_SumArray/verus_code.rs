use vstd::prelude::*;

verus! {
    spec fn sum_spec(s: Seq<i32>, len: int) -> int
        recommends 0 <= len <= s.len()
        decreases len
    {
        if len <= 0 { 
            0 
        } else { 
            s[len - 1] as int + sum_spec(s, len - 1) 
        }
    }

    fn sum_array(arr: &Vec<i32>) -> (result: i32)
        requires 
            arr.len() > 0,
            arr.len() <= 5,
            forall|j: int| 0 <= j < arr.len() ==> #[trigger] arr[j] >= -5 && arr[j] <= 5,
        ensures result as int == sum_spec(arr@, arr.len() as int)
    {
        let mut sum = 0i32;
        let mut i: usize = 0;
        while i < arr.len()
            invariant 
                0 <= i <= arr.len(),
                sum as int == sum_spec(arr@, i as int),
                -25 <= sum as int <= 25,
            decreases arr.len() - i
        {
            sum = sum + arr[i];
            i = i + 1;
        }
        sum
    }
}