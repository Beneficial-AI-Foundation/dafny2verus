use vstd::prelude::*;

verus! {
    // Predicate for strictly sorted sequences
    spec fn strict_sorted(s: Seq<int>) -> bool {
        forall|u: int, w: int| 0 <= u < w < s.len() ==> s[u] < s[w]
    }

    // Helper function to check if an element is in a slice prefix
    spec fn in_prefix(val: int, arr: Seq<int>, len: int) -> bool {
        exists|j: int| 0 <= j < len && arr[j] == val
    }

    // Method to check if first n elements of v are contained in first m elements of w
    // This is an O(m+n) algorithm for strictly increasing ordered arrays
    fn mcontained(v: &[int], w: &[int], n: usize, m: usize) -> (b: bool)
        requires
            n <= m,
            strict_sorted(v@),
            strict_sorted(w@),
            v.len() >= n && w.len() >= m,
        ensures
            b == forall|k: int| #![auto] 0 <= k < n ==> in_prefix(v@[k], w@, m as int)
    {
        let mut i: usize = 0;
        let mut j: usize = 0;
        
        // Two-pointer approach: advance through both arrays
        while i < n && j < m && v[i] >= w[j]
            invariant
                0 <= i <= n,
                0 <= j <= m,
                v.len() >= n,
                w.len() >= m,
                strict_sorted(v@),
                strict_sorted(w@),
                // All elements v[0..i] are contained in w[0..j]
                forall|k: int| #![auto] 0 <= k < i ==> in_prefix(v@[k], w@, j as int),
                // Current element v[i] is not in w[0..j] (if i < n)
                i < n ==> !in_prefix(v@[i as int], w@, j as int),
            decreases w.len() - j, v.len() - i
        {
            if v[i] == w[j] {
                // Found v[i] in w, move to next element in v
                i = i + 1;
            }
            // Always advance j to continue searching in w
            j = j + 1;
        }
        
        proof {
            // Key insight: if we exit the loop with i < n, then v[i] cannot be in w[..m]
            // because w is strictly sorted and we've checked all elements in w[..j] that are <= v[i]
            if i < n {
                assert(!in_prefix(v@[i as int], w@, m as int));
            }
        }
        
        // Return true iff all n elements of v have been found in w
        i == n
    }
}