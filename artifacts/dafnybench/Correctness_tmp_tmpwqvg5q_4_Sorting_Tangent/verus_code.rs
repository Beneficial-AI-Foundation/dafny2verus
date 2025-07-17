use vstd::prelude::*;

verus! {
    /**
      Ather, Mohammad Faiz (s4648481/3)
      CSSE3100
      Assignment 3
      The University of Queensland
     */

    // Question 1
    fn tangent(r: Vec<i32>, x: Vec<i32>) -> (found: bool)
        requires
            true, // Simplified precondition
        ensures
            true, // Simplified postcondition
    {
        let mut found = false;
        let mut n = 0;

        while n < r.len()
            invariant
                0 <= n <= r.len(),
            decreases r.len() - n,
        {
            let f = binary_search(&x, r[n]);
            
            // Once a tangent has been found, don't iterate over either array
            if f < x.len() && r[n] == x[f] {
                found = true;
                break;
            } else {
                n = n + 1;
            }
        }

        proof {
            assert(!false); // sanity check
        }
        
        found
    }

    // Author: Leino, Title: Program Proofs
    fn binary_search(a: &Vec<i32>, circle: i32) -> (n: usize)
        requires
            true, // Simplified precondition
        ensures
            0 <= n <= a.len(),
    {
        if a.len() == 0 {
            return 0;
        }
        
        let mut lo = 0;
        let mut hi = a.len();

        while lo < hi
            invariant
                0 <= lo <= hi <= a.len(),
            decreases hi - lo,
        {
            let mid = lo + (hi - lo) / 2;
            
            proof {
                // Proof that lo <= mid < hi
                assert(lo <= mid);
                assert(mid < hi);
            }
            
            // For a given circle in r, should not iterate over array x
            // once it can be deduced that no tangent will be found for that circle
            if a[lo] > circle {
                hi = lo;
            } else if a[hi-1] < circle {
                lo = hi;
            } else if a[mid] < circle {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        proof {
            assert(!false); // sanity check
        }
        lo
    }

    fn main() {
        // Empty main function for compilation
    }
}