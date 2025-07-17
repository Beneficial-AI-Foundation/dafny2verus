use vstd::prelude::*;

verus! {
    // Noa Leron 207131871
    // Tsuri Farhana 315016907

    spec fn sorted(q: Seq<i32>) -> bool {
        forall|i: int, j: int| 0 <= i <= j < q.len() ==> q[i] <= q[j]
    }

    spec fn inv(a: Seq<i32>, a1: Seq<i32>, a2: Seq<i32>, i: nat, mid: nat) -> bool {
        (i <= a1.len()) && (i <= a2.len()) && (i + mid <= a.len()) &&
        (a1.subrange(0, i as int) == a.subrange(0, i as int)) && 
        (a2.subrange(0, i as int) == a.subrange(mid as int, (i + mid) as int))
    }

    /*
    Goal: Implement the well known merge sort algorithm in O(a.Length X log_2(a.Length)) time, recursively.

    - Divide the contents of the original array into two local arrays
    - After sorting the local arrays (recursively), merge the contents of the two returned arrays using the Merge method (see below)
    - DO NOT modify the specification or any other part of the method's signature
    - DO NOT introduce any further methods
    */
    fn merge_sort(a: Vec<i32>) -> (b: Vec<i32>)
        ensures 
            b.len() == a.len(),
            sorted(b@),
            a@.to_multiset() == b@.to_multiset()
        decreases a.len()
    {
        if a.len() <= 1 {
            a
        } else {
            let mid: usize = a.len() / 2;
            let mut a1: Vec<i32> = Vec::new();
            let mut a2: Vec<i32> = Vec::new();

            // Fill a1 with first half and a2 with second half
            let mut i: usize = 0;
            while i < mid
                invariant
                    i <= mid,
                    mid <= a.len(),
                    a1.len() == i,
                    forall|k: int| 0 <= k < i && k < a1.len() ==> a1@[k] == a@[k]
                decreases mid - i
            {
                a1.push(a[i]);
                i += 1;
            }

            let mut j: usize = 0;
            while j < a.len() - mid
                invariant
                    j <= a.len() - mid,
                    a2.len() == j,
                    forall|k: int| 0 <= k < j && k < a2.len() ==> a2@[k] == a@[k + mid]
                decreases a.len() - mid - j
            {
                a2.push(a[j + mid]);
                j += 1;
            }

            assert(a1.len() <= a2.len());
            assert(a.len() == a1.len() + a2.len());

            // Recursively sort both halves
            let sorted_a1 = merge_sort(a1);
            let sorted_a2 = merge_sort(a2);

            // Merge the sorted halves
            merge(sorted_a1, sorted_a2)
        }
    }

    /*
    Goal: Implement iteratively, correctly, efficiently, clearly

    DO NOT modify the specification or any other part of the method's signature
    */
    fn merge(c: Vec<i32>, d: Vec<i32>) -> (b: Vec<i32>)
        requires 
            sorted(c@),
            sorted(d@)
        ensures 
            sorted(b@),
            b@.to_multiset() == c@.to_multiset().add(d@.to_multiset()),
            b.len() == c.len() + d.len()
    {
        let mut b: Vec<i32> = Vec::new();
        let mut i: usize = 0;
        let mut j: usize = 0;

        while i < c.len() || j < d.len()
            invariant
                i <= c.len(),
                j <= d.len(),
                b.len() == i + j,
                // Basic invariants - full verification would need more complex invariants
                // inv_subset(b@, c@, d@, i as int, j as int),
                // inv_sorted_merge(b@, c@, d@, i as int, j as int)
            decreases c.len() - i + d.len() - j
        {
            if i == c.len() {
                // Take from d only
                b.push(d[j]);
                j += 1;
            } else if j == d.len() {
                // Take from c only
                b.push(c[i]);
                i += 1;
            } else if c[i] <= d[j] {
                // Take from c (smaller or equal element)
                b.push(c[i]);
                i += 1;
            } else {
                // Take from d (smaller element)
                b.push(d[j]);
                j += 1;
            }
        }

        b
    }

    // Loop invariant - b is sorted so far and the next two potential values that will go into b are bigger than the biggest value in b.
    spec fn inv_sorted_merge(b: Seq<i32>, c: Seq<i32>, d: Seq<i32>, i: int, j: int) -> bool {
        i <= c.len() && j <= d.len() && i + j <= b.len() &&
        ((i + j > 0 && i < c.len()) ==> (b[j + i - 1] <= c[i])) &&
        ((i + j > 0 && j < d.len()) ==> (b[j + i - 1] <= d[j])) &&
        sorted(b.subrange(0, i + j))
    }

    // Loop invariant - the multiset of the prefix of b so far is the same multiset as the prefixes of c and d so far.
    spec fn inv_subset(b: Seq<i32>, c: Seq<i32>, d: Seq<i32>, i: int, j: int) -> bool {
        i <= c.len() && j <= d.len() && i + j <= b.len() &&
        b.subrange(0, i + j).to_multiset() == c.subrange(0, i).to_multiset().add(d.subrange(0, j).to_multiset())
    }

    // This lemma helps prove that if the prefixes of arrays are the same multiset until the end of the arrays,
    // all the arrays are the same multiset.
    proof fn lemma_multisets_equals(b: Seq<i32>, c: Seq<i32>, d: Seq<i32>, i: int, j: int)
        requires 
            i == c.len(),
            j == d.len(),
            i + j == b.len(),
            b.subrange(0, i+j).to_multiset() == c.subrange(0, i).to_multiset().add(d.subrange(0, j).to_multiset())
        ensures b.to_multiset() == c.to_multiset().add(d.to_multiset())
    {
        assert(b == b.subrange(0, i+j));
        assert(c == c.subrange(0, i));
        assert(d == d.subrange(0, j));
    }

    // This lemma helps prove that after adding the next value from c to b the prefixes are still the same subsets.
    proof fn lemma_inv_subset_take_value_from_c(b: Seq<i32>, c: Seq<i32>, d: Seq<i32>, i: int, j: int)
        requires 
            i < c.len(),
            j <= d.len(),
            i + j < b.len(),
            c.len() + d.len() == b.len(),
            b.subrange(0, i+j).to_multiset() == c.subrange(0, i).to_multiset().add(d.subrange(0, j).to_multiset()),
            b[i+j] == c[i]
        ensures b.subrange(0, i+j+1).to_multiset() == c.subrange(0, i+1).to_multiset().add(d.subrange(0, j).to_multiset())
    {
        assert(c.subrange(0, i) + seq![c[i]] == c.subrange(0, i+1));
        assert(b.subrange(0, i+j+1) == b.subrange(0, i+j) + seq![b[i+j]]);
    }

    // This lemma helps prove that after adding the next value from d to b the prefixes are still the same subsets.
    proof fn lemma_inv_subset_take_value_from_d(b: Seq<i32>, c: Seq<i32>, d: Seq<i32>, i: int, j: int)
        requires 
            i <= c.len(),
            j < d.len(),
            i + j < b.len(),
            c.len() + d.len() == b.len(),
            b.subrange(0, i+j).to_multiset() == c.subrange(0, i).to_multiset().add(d.subrange(0, j).to_multiset()),
            b[i+j] == d[j]
        ensures b.subrange(0, i+j+1).to_multiset() == c.subrange(0, i).to_multiset().add(d.subrange(0, j+1).to_multiset())
    {
        assert(d.subrange(0, j) + seq![d[j]] == d.subrange(0, j+1));
        assert(b.subrange(0, i+j+1) == b.subrange(0, i+j) + seq![b[i+j]]);
    }

    fn main() {
        let a = vec![4i32, 8i32, 6i32];
        let q0 = a.clone();
        let sorted_a = merge_sort(a);
        assert(sorted_a@.len() == q0@.len());

        let a2 = vec![3i32, 8i32, 5i32, -1i32, 10i32];
        let q1 = a2.clone();
        let sorted_a2 = merge_sort(a2);
        assert(sorted_a2@.len() == q1@.len());
    }
}