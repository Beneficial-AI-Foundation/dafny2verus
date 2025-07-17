use vstd::prelude::*;

verus! {
    /// Translated from Dafny SlopeSearch method
    /// Searches for a key in a 2D array that is sorted both row-wise and column-wise
    fn slope_search(a: &Vec<Vec<i32>>, key: i32) -> (result: (usize, usize))
        requires 
            a.len() > 0,
            forall|i: int| 0 <= i < a.len() ==> #[trigger] a@[i].len() > 0,
            forall|i: int| 0 <= i < a.len() ==> #[trigger] a@[i].len() == a@[0].len(),
            a@[0].len() > 0,
            // Row-wise sorted: each row is sorted in ascending order
            forall|i: int, j: int, j_prime: int| #![auto]
                0 <= i < a.len() && 0 <= j < j_prime < a@[i].len() ==> 
                a@[i]@[j] <= a@[i]@[j_prime],
            // Column-wise sorted: each column is sorted in ascending order
            forall|i: int, i_prime: int, j: int| #![auto]
                0 <= i < i_prime < a.len() && 0 <= j < a@[i].len() ==> 
                a@[i]@[j] <= a@[i_prime]@[j],
            // Key exists in the array
            exists|i: int, j: int| #![auto]
                0 <= i < a.len() && 0 <= j < a@[i].len() && 
                a@[i]@[j] == key,
        ensures
            result.0 < a.len(),
            result.1 < a@[result.0 as int].len(),
            a@[result.0 as int]@[result.1 as int] == key,
    {
        // Start at top-right corner (same as original Dafny)
        let mut m: usize = 0;
        let mut n: usize = a[0].len() - 1;
        
        while a[m][n] != key && m < a.len() - 1 && n > 0
            invariant
                m < a.len(),
                n < a@[0].len(),
                forall|i: int| 0 <= i < a.len() ==> #[trigger] a@[i].len() == a@[0].len(),
                n < a@[m as int].len(),
            decreases (a.len() - m) + n,
        {
            if a[m][n] < key {
                m = m + 1;
            } else {
                n = n - 1;
            }
        }
        
        (m, n)
    }
}