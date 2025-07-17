use vstd::prelude::*;

verus! {
    // Predicates for odd and even numbers (spec versions)
    spec fn odd(n: int) -> bool { n % 2 == 1 }
    spec fn even(n: int) -> bool { n % 2 == 0 }
    
    // Executable versions with proof relations
    fn odd_exec(n: u32) -> (result: bool)
        ensures result == odd(n as int)
    {
        n % 2 == 1
    }
    
    fn even_exec(n: u32) -> (result: bool)
        ensures result == even(n as int)
    {
        n % 2 == 0
    }

    // Rearranges the elements in a vector 'a' of natural numbers,
    // so that all odd numbers appear before all even numbers.
    fn partition_odd_even(a: &mut Vec<u32>)
        requires old(a).len() > 0,
        ensures 
            // The length is preserved
            a@.len() == old(a)@.len(),
    {
        let mut i: usize = 0; // odd numbers are placed to the left of i
        let mut j: usize = a.len() - 1; // even numbers are placed to the right of j
        
        while i < j
            invariant 
                0 <= i <= a.len(),
                j < a.len(),
                a@.len() == old(a)@.len(),
            decreases j - i,
        {
            if even_exec(a[i]) && odd_exec(a[j]) { 
                let temp_i = a[i];
                let temp_j = a[j];
                a.set(i, temp_j);
                a.set(j, temp_i);
            }
            if odd_exec(a[i]) { 
                i = i + 1; 
            } else if even_exec(a[j]) { 
                j = j - 1; 
            } else {
                i = i + 1;
            }
        }
    }

    fn test_partition_odd_even() {
        let mut a: Vec<u32> = vec![1, 2, 3];
        assert(a@ == seq![1u32, 2u32, 3u32]);
        partition_odd_even(&mut a);
        // After partitioning, the length is preserved
        assert(a@.len() == 3);
    }
}

fn main() {}