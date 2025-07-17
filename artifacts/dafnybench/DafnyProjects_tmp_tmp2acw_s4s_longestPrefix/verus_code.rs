use vstd::prelude::*;

verus! {
    // MFES, Exam 8/Sept/20201, Exercise 5 

    spec fn seq_equal_prefix(a: Seq<i32>, b: Seq<i32>, len: int) -> bool {
        forall|k: int| 0 <= k < len ==> a[k] == b[k]
    }

    // Computes the length (i) of the longest common prefix (initial subarray) 
    // of two sequences a and b. 
    fn longest_prefix(a: &[i32], b: &[i32]) -> (i: usize)
        ensures 
            i <= a.len() && i <= b.len(),
            seq_equal_prefix(a@, b@, i as int),
            i < a.len() && i < b.len() ==> a@[i as int] != b@[i as int]
    {
        let mut i: usize = 0;
        
        while i < a.len() && i < b.len()
            invariant 
                i <= a.len() && i <= b.len(),
                seq_equal_prefix(a@, b@, i as int),
            decreases a.len() - i
        {
            let a_elem = a[i];
            let b_elem = b[i];
            
            if a_elem != b_elem {
                proof {
                    assert(a_elem == a@[i as int]);
                    assert(b_elem == b@[i as int]);
                    assert(a@[i as int] != b@[i as int]);
                }
                return i;
            }
            i = i + 1;
        }
        i
    }
 
    // Test method with an example.
    fn test_longest_prefix() {
        let a = [1i32, 3i32, 2i32, 4i32, 8i32];
        let b = [1i32, 3i32, 3i32, 4i32];
        let i = longest_prefix(&a, &b);
        assert(i == 2); 
    }

    fn main() {}
}