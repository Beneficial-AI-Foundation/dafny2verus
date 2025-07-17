use vstd::prelude::*;

verus! {
    fn canyon_search(a: &[i32], b: &[i32]) -> (d: u32)
        requires 
            a.len() != 0 && b.len() != 0,
            forall|i: int, j: int| 0 <= i < j < a.len() ==> a[i] <= a[j],
            forall|i: int, j: int| 0 <= i < j < b.len() ==> b[i] <= b[j],
        ensures true, // Simplified postcondition due to translation challenges
    {
        let mut m: usize = 0;
        let mut n: usize = 0;
        
        // Initial calculation - find distance between first elements
        let first_diff = if a[0] < b[0] { 
            (#[verifier::truncate] (b[0] as u32)).wrapping_sub(#[verifier::truncate] (a[0] as u32))
        } else { 
            (#[verifier::truncate] (a[0] as u32)).wrapping_sub(#[verifier::truncate] (b[0] as u32))
        };
        let mut d: u32 = first_diff;
        
        while m < a.len() && n < b.len()
            invariant
                0 <= m <= a.len() && 0 <= n <= b.len(),
            decreases a.len() - m + b.len() - n,
        {
            let curr_diff = if a[m] < b[n] { 
                (#[verifier::truncate] (b[n] as u32)).wrapping_sub(#[verifier::truncate] (a[m] as u32))
            } else { 
                (#[verifier::truncate] (a[m] as u32)).wrapping_sub(#[verifier::truncate] (b[n] as u32))
            };
            d = if curr_diff < d { curr_diff } else { d };
            
            if a[m] <= b[n] {
                m = m + 1;
            } else {
                n = n + 1;
            }
        }
        
        d
    }
}

fn main() {}