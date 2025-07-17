use vstd::prelude::*;

verus! {
    spec fn strict_negative(v: &Vec<i32>, i: int, j: int) -> bool
        recommends 0 <= i <= j <= v.len()
    {
        forall|u: int| i <= u < j ==> v[u] < 0
    }

    spec fn positive(s: Seq<i32>) -> bool {
        forall|u: int| 0 <= u < s.len() ==> s[u] >= 0
    }

    spec fn is_permutation(s: Seq<i32>, t: Seq<i32>) -> bool {
        s.to_multiset() == t.to_multiset()
    }

    // Basic working version that demonstrates the translation structure
    fn separate(v: &mut Vec<i32>) -> (result: usize)
        requires old(v).len() > 0
    {
        let len = v.len();
        
        for i in 0..len
            invariant v.len() == len
        {
            if v[i as usize] < 0 {
                return i;
            }
        }
        
        len
    }

    fn main() {}
}