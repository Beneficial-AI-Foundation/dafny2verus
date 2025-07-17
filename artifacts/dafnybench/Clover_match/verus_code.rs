use vstd::prelude::*;

verus! {
    fn match_strings(s: Vec<char>, p: Vec<char>) -> (b: bool)
        requires s.len() == p.len(),
        ensures b <==> forall|n: int| 0 <= n < s.len() ==> 
            s[n] == p[n] || p[n] == '?'
    {
        let mut i: usize = 0;
        while i < s.len()
            invariant 
                0 <= i <= s.len(),
                s.len() == p.len(),
                forall|n: int| 0 <= n < i ==> 
                    s[n] == p[n] || p[n] == '?'
            decreases s.len() - i
        {
            assert(i < s.len());
            assert(i < p.len());
            if s[i] != p[i] && p[i] != '?' {
                return false;
            }
            i = i + 1;
        }
        true
    }
}

fn main() {}