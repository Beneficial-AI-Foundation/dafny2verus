use vstd::prelude::*;

verus! {
    // Insertion sort.
    //
    // Author: Snorri Agnarsson, snorri@hi.is
    // Translated to Verus

    spec fn is_sorted(s: Seq<int>) -> bool {
        forall|p: int, q: int| 0 <= p < q < s.len() ==> s[p] <= s[q]
    }

    fn insertion_sort(s: &Vec<int>) -> (r: Vec<int>)
        ensures 
            s@.to_multiset() == r@.to_multiset(),
            is_sorted(r@),
    {
        let mut r: Vec<int> = Vec::new();
        let mut i = 0;
        
        while i < s.len()
            invariant 
                0 <= i <= s.len(),
                s@.subrange(0, i as int).to_multiset() == r@.to_multiset(),
                is_sorted(r@),
            decreases s.len() - i,
        {
            let x = s[i];
            
            // Find insertion position
            let mut k = r.len();
            while k > 0
                invariant 
                    0 <= k <= r.len(),
                    forall|p: int| k <= p < r.len() ==> r@[p] > x,
                decreases k,
            {
                if r[k - 1] <= x {
                    break;
                }
                k = k - 1;
            }
            
            // Insert x at position k
            r.insert(k, x);
            i = i + 1;
        }
        
        r
    }
}

fn main() {}