use vstd::prelude::*;

verus! {
    fn fillK(a: &[int], n: usize, k: int, c: usize) -> (b: bool)
        requires 
            c <= n,
            n == a.len(),
        ensures true,
    {
        if c == 0 {
            return true;
        }

        let mut p = 0;
        while p < c
            invariant 0 <= p <= c,
            decreases c - p,
        {
            if p < a.len() && a[p] != k {
                return false;
            }
            p = p + 1;
        }
        true
    }

    fn containsSubString(a: &[char], b: &[char]) -> (pos: isize)
        requires 
            b.len() <= a.len(),
        ensures true,
    {
        let mut pos: isize = -1;

        if b.len() == 0 {
            return pos;
        }

        let mut p = 0;

        while p < a.len()
            invariant 0 <= p <= a.len(),
            decreases a.len() - p,
        {
            if a.len() - p < b.len() {
                return pos;
            }

            if p < a.len() && b.len() > 0 && a[p] == b[0] {
                let mut i = 0;
                while i < b.len()
                    invariant 
                        0 <= i <= b.len(),
                        p < a.len(),
                        a.len() - p >= b.len(),
                    decreases b.len() - i,
                {
                    if i + p < a.len() && i < b.len() && a[i + p] != b[i] {
                        return -1;
                    }
                    i = i + 1;
                }
                pos = #[verifier::truncate] (p as isize);
                return pos;
            }

            p = p + 1;
        }
        pos
    }
}

fn main() {}