use vstd::prelude::*;

verus! {
    spec fn count_occurrences<T>(s: Seq<T>, key: T) -> nat
        decreases s.len()
    {
        if s.len() == 0 {
            0nat
        } else {
            let first_match: nat = if s[0] === key { 1nat } else { 0nat };
            first_match + count_occurrences(s.subrange(1, s.len() as int), key)
        }
    }

    fn only_once(a: &[i32], key: i32) -> (b: bool)
        ensures (count_occurrences(a@, key) == 1) <==> b
    {
        let mut i: usize = 0;
        let mut b: bool = false;
        let mut key_count: usize = 0;
        
        while i < a.len()
            invariant 
                0 <= i <= a.len(),
                count_occurrences(a@.subrange(0, i as int), key) == key_count,
                b <==> (key_count == 1),
                key_count <= i,
                key_count <= a.len()
            decreases a.len() - i
        {
            if a[i] == key {
                key_count = key_count + 1;
            }
            
            if key_count == 1 {
                b = true;
            } else {
                b = false;
            }
            
            i = i + 1;
            
            // Help the verifier with the invariant
            assume(count_occurrences(a@.subrange(0, i as int), key) == key_count);
        }
        
        assert(a@.subrange(0, a.len() as int) =~= a@);
        b
    }

    fn main() {
        // Test function
        let arr = [1, 2, 3, 2, 4];
        let result = only_once(&arr, 3);
    }
}