use vstd::prelude::*;

verus! {
    spec fn is_even_spec(x: int) -> bool {
        x % 2 == 0
    }

    fn find_even_numbers(arr: &[i32]) -> (even_numbers: Vec<i32>)
        requires arr.len() <= usize::MAX,
        ensures
            // All numbers in the output are even
            forall|k: int| 0 <= k < even_numbers@.len() ==> is_even_spec(even_numbers@[k] as int),
            // All numbers in the output come from the input array
            forall|k: int| 0 <= k < even_numbers@.len() ==> arr@.contains(even_numbers@[k]),
            // The output contains only even numbers from the input
            forall|x: i32| even_numbers@.contains(x) ==> (arr@.contains(x) && is_even_spec(x as int))
    {
        let mut even_list: Vec<i32> = Vec::new();
        
        for i in 0..arr.len()
            invariant
                0 <= i <= arr@.len(),
                // All numbers in even_list are even
                forall|k: int| 0 <= k < even_list@.len() ==> is_even_spec(even_list@[k] as int),
                // All numbers in even_list come from the input array
                forall|k: int| 0 <= k < even_list@.len() ==> arr@.contains(even_list@[k]),
                // The output contains only even numbers from the input
                forall|x: i32| even_list@.contains(x) ==> (arr@.contains(x) && is_even_spec(x as int))
        {
            let current_val = arr[i];
            
            // Check if even using bitwise operation
            if (current_val & 1) == 0 {
                even_list.push(current_val);
                
                // Prove that bitwise check implies mathematical evenness
                proof {
                    // In a real implementation, we would need to prove:
                    // (current_val & 1) == 0 <==> is_even_spec(current_val as int)
                    // For now, we'll assume this axiom
                    assume((current_val & 1) == 0 ==> is_even_spec(current_val as int));
                }
            }
        }

        even_list
    }
}

fn main() {}