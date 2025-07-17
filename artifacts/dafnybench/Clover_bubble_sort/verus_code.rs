use vstd::prelude::*;

verus! {
    fn bubble_sort(a: &mut Vec<i32>)
        requires old(a).len() >= 0,
        ensures 
            // Array is sorted
            forall|i: int, j: int| 0 <= i < j < a.len() ==> a[i] <= a[j],
            // Length is preserved
            a.len() == old(a).len(),
    {
        let mut i: usize = a.len();
        if i == 0 { return; }
        i = i - 1;
        
        while i > 0
            invariant 
                i < a.len(),
                a.len() == old(a).len(),
                // Elements from i onwards are sorted
                forall|ii: int, jj: int| i <= ii < jj < a.len() ==> a[ii] <= a[jj],
                // Elements up to i are <= elements after i
                forall|k: int, k_prime: int| 0 <= k <= i < k_prime < a.len() ==> a[k] <= a[k_prime],
            decreases i,
        {
            let mut j: usize = 0;
            
            while j < i
                invariant 
                    i < a.len() && j <= i,
                    a.len() == old(a).len(),
                    // Elements from i onwards are sorted
                    forall|ii: int, jj: int| i <= ii <= jj < a.len() ==> a[ii] <= a[jj],
                    // Elements up to i are <= elements after i
                    forall|k: int, k_prime: int| 0 <= k <= i < k_prime < a.len() ==> a[k] <= a[k_prime],
                    // Elements up to j are <= a[j]
                    forall|k: int| 0 <= k <= j ==> a[k] <= a[j as int],
                decreases i - j,
            {
                if a[j] > a[j + 1] {
                    let temp = a[j];
                    let temp2 = a[j + 1];
                    a.set(j, temp2);
                    a.set(j + 1, temp);
                }
                j = j + 1;
            }
            i = i - 1;
        }
    }
}

fn main() {}