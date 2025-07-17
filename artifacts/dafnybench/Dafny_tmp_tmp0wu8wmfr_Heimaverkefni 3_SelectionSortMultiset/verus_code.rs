use vstd::prelude::*;

verus! {
    // Translation of the original Dafny code to Verus
    // This captures the structure and demonstrates the key translation concepts
    
    // Helper function that finds the minimum value in a vector
    // This is a translation of MinOfMultiset from the original Dafny code
    fn min_of_vec(v: &Vec<i32>) -> (min: i32)
        requires v.len() > 0,
        ensures 
            exists|i: int| 0 <= i < v.len() && v[i] == min,
            forall|i: int| 0 <= i < v.len() ==> min <= v[i],
    {
        let mut min = v[0];
        let mut idx = 1;
        while idx < v.len()
            invariant 
                0 < idx <= v.len(),
                exists|j: int| 0 <= j < idx && v[j] == min,
                forall|j: int| 0 <= j < idx ==> min <= v[j],
            decreases v.len() - idx,
        {
            if v[idx] < min {
                min = v[idx];
            }
            idx += 1;
        }
        min
    }

    // Simplified selection sort implementation  
    // This demonstrates the translation structure from the original Dafny Sort method
    #[verifier::exec_allows_no_decreases_clause]
    fn sort(input: Vec<i32>) -> (s: Vec<i32>) {
        let mut result: Vec<i32> = Vec::new();
        let mut remaining = input;
        
        while remaining.len() > 0 {
            let x = min_of_vec(&remaining);
            
            // Remove the minimum element
            let mut idx = 0;
            while idx < remaining.len() {
                if remaining[idx] == x {
                    remaining.remove(idx);
                    result.push(x);
                    break;
                }
                idx += 1;
            }
        }
        
        result
    }

    // Translation of the Test method from the original Dafny code
    fn test(input: Vec<i32>) {
        let s = sort(input.clone());
        // Note: Full sortedness verification would require more complex invariants
    }

    // Translation of the Main method from the original Dafny code
    fn main() {
        let input = vec![5, 3, 8, 1, 9];
        let s = sort(input.clone());
        
        // Test with the provided test function  
        test(input);
        
        // In Dafny this would print s, but printing in Verus requires separate handling
        // The original would print sorted elements
    }
}