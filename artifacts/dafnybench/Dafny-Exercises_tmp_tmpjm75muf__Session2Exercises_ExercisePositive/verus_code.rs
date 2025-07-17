use vstd::prelude::*;

verus! {
    // Predicate to check if all elements in a sequence are positive (non-negative)
    spec fn positive(s: Seq<i32>) -> bool {
        forall|u: int| 0 <= u < s.len() ==> s[u] >= 0
    }

    // Method to check if all elements in an array are positive
    fn mpositive(v: &[i32]) -> (b: bool)
        ensures b == positive(v@)
    {
        let mut i: usize = 0;
        
        while i < v.len() && v[i] >= 0
            invariant 
                0 <= i <= v.len(),
                forall|u: int| 0 <= u < i ==> v@[u] >= 0,
            decreases v.len() - i
        {
            i = i + 1;
        }
        
        let result = i == v.len();
        
        // Post-loop analysis
        if result {
            // If we processed all elements, then all are positive
            assert forall|u: int| 0 <= u < v@.len() implies v@[u] >= 0 by {
                assert(i == v.len());
                if 0 <= u < v@.len() {
                    assert(u < i);
                    assert(v@[u] >= 0);
                }
            };
        } else {
            // If we didn't process all elements, then we found a negative one
            assert(i < v.len());
            assert(v@[i as int] < 0);
            assert(!positive(v@));
        }
        
        result
    }

    // Alternative implementation using boolean flag  
    fn mpositive3(v: &[i32]) -> (b: bool)
        ensures b == positive(v@)
    {
        let mut i: usize = 0;
        let mut b: bool = true;
        
        while i < v.len() && b
            invariant 
                0 <= i <= v.len(),
                b == (forall|u: int| 0 <= u < i ==> v@[u] >= 0),
                !b ==> !positive(v@)
            decreases v.len() - i
        {
            b = v[i] >= 0;
            i = i + 1;
        }
        
        if i == v.len() && b {
            assert forall|u: int| 0 <= u < v@.len() implies v@[u] >= 0 by {
                if 0 <= u < v@.len() {
                    assert(u < i);
                    assert(v@[u] >= 0);
                }
            };
        }
        
        b
    }

    // Method mpositive4 (identical to mpositive3)
    fn mpositive4(v: &[i32]) -> (b: bool)
        ensures b == positive(v@)
    {
        let mut i: usize = 0;
        let mut b: bool = true;
        
        while i < v.len() && b
            invariant 
                0 <= i <= v.len(),
                b == (forall|u: int| 0 <= u < i ==> v@[u] >= 0),
                !b ==> !positive(v@)
            decreases v.len() - i
        {
            b = v[i] >= 0;
            i = i + 1;
        }
        
        if i == v.len() && b {
            assert forall|u: int| 0 <= u < v@.len() implies v@[u] >= 0 by {
                if 0 <= u < v@.len() {
                    assert(u < i);
                    assert(v@[u] >= 0);
                }
            };
        }
        
        b
    }

    // Right-to-left traversal implementation - simplified
    fn mpositivertl(v: &[i32]) -> (b: bool)
        ensures b == positive(v@)
    {
        if v.len() == 0 {
            return true;
        }
        
        // Check elements from right to left
        for i in 0..v.len()
            invariant forall|u: int| v.len() - i <= u < v@.len() ==> v@[u] >= 0
        {
            let idx = v.len() - 1 - i;
            if v[idx] < 0 {
                assert(v@[idx as int] < 0);
                assert(!positive(v@));
                return false;
            }
        }
        
        // All elements are positive
        assert forall|u: int| 0 <= u < v@.len() implies v@[u] >= 0 by {
            if 0 <= u < v@.len() {
                assert(v.len() - v.len() <= u);
                assert(v@[u] >= 0);
            }
        };
        
        true
    }
}

fn main() {}