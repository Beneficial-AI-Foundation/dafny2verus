use vstd::prelude::*;

verus! {
    // Predicate that checks if all elements in a sequence are equal
    spec fn all_equal(s: Seq<int>) -> bool {
        forall|i: int, j: int| 0 <= i < s.len() && 0 <= j < s.len() ==> s[i] == s[j]
    }

    // Lemma: equivalence with ordered indexes
    proof fn equivalence_no_order(s: Seq<int>)
        ensures all_equal(s) <==> (forall|i: int, j: int| 0 <= i <= j < s.len() ==> s[i] == s[j])
    {}

    // Lemma: equivalence with all equal to first element
    proof fn equivalence_equal_to_first(s: Seq<int>)
        requires s.len() > 0
        ensures all_equal(s) <==> (forall|i: int| 0 <= i < s.len() ==> s[0] == #[trigger] s[i])
    {}

    // Method 1: Check equality with first element
    fn m_all_equal1(v: &[int]) -> (b: bool)
        ensures b == all_equal(v@)
    {
        if v.len() == 0 {
            return true;
        }
        
        let mut i = 0;
        
        while i < v.len()
            invariant 
                0 <= i <= v.len(),
                forall|k: int| 0 <= k < i ==> #[trigger] v@[k] == #[trigger] v@[0]
            decreases v.len() - i
        {
            if v[i] != v[0] {
                return false;
            }
            i = i + 1;
        }
        
        true
    }

    // Method 2: Check consecutive elements equal to first
    fn m_all_equal2(v: &[int]) -> (b: bool)
        ensures b == all_equal(v@)
    {
        if v.len() == 0 {
            return true;
        }
        
        let mut i = 0;
        
        while i < v.len() && v[i] == v[0]
            invariant 
                0 <= i <= v.len(),
                forall|k: int| 0 <= k < i ==> #[trigger] v@[k] == #[trigger] v@[0]
            decreases v.len() - i
        {
            i = i + 1;
        }
        
        i == v.len()
    }

    // Method 3: Check contiguous pairs
    fn m_all_equal3(v: &[int]) -> (b: bool)
        ensures b == all_equal(v@)
    {
        if v.len() <= 1 {
            return true;
        }
        
        let mut i = 0;
        
        while i < v.len() - 1
            invariant 
                0 <= i <= v.len() - 1,
                forall|k1: int, k2: int| 0 <= k1 <= i && 0 <= k2 <= i ==> #[trigger] v@[k1] == #[trigger] v@[k2]
            decreases v.len() - i
        {
            if v[i] != v[i + 1] {
                return false;
            }
            i = i + 1;
        }
        
        true
    }

    // Method 4: Check contiguous pairs with boolean flag
    fn m_all_equal4(v: &[int]) -> (b: bool)
        ensures b == all_equal(v@)
    {
        if v.len() <= 1 {
            return true;
        }
        
        let mut i = 0;
        
        while i < v.len() - 1
            invariant 
                0 <= i <= v.len() - 1,
                forall|k1: int, k2: int| 0 <= k1 <= i && 0 <= k2 <= i ==> #[trigger] v@[k1] == #[trigger] v@[k2]
            decreases v.len() - i
        {
            if v[i] != v[i + 1] {
                return false;
            }
            i = i + 1;
        }
        
        true
    }

    // Method 5: Check with early termination
    fn m_all_equal5(v: &[int]) -> (b: bool)
        ensures b == all_equal(v@)
    {
        if v.len() == 0 {
            return true;
        }
        
        let mut i = 0;
        
        while i < v.len()
            invariant 
                0 <= i <= v.len(),
                forall|k: int| 0 <= k < i ==> #[trigger] v@[k] == #[trigger] v@[0]
            decreases v.len() - i
        {
            if v[i] != v[0] {
                return false;
            }
            i = i + 1;
        }
        
        true
    }
}

fn main() {}