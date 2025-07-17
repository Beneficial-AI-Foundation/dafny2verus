use vstd::prelude::*;

verus! {
    spec fn sorted(s: Seq<int>) -> bool {
        forall|u: int, w: int| 0 <= u < w < s.len() ==> s[u] <= s[w]
    }

    spec fn as_int_seq(s: Seq<i32>) -> Seq<int> {
        s.map(|_i, x| x as int)
    }

    // Binary search returns index of last element <= elem, or -1 if none
    fn binary_search(v: &[i32], elem: i32) -> (p: i32)
        requires sorted(as_int_seq(v@))
        ensures 
            -1 <= p < v@.len() &&
            (forall|u: int| 0 <= u <= p ==> v@[u] <= elem) && 
            (forall|w: int| p < w < v@.len() ==> v@[w] > elem)
    {
        let mut c: usize = 0;
        let mut f: usize = if v.len() == 0 { 0 } else { v.len() - 1 };
        
        // Handle empty array case
        if v.len() == 0 {
            return -1;
        }
        
        while c <= f
            invariant 
                c <= v@.len() && 
                f < v@.len() && 
                (forall|u: int| 0 <= u < c ==> v@[u] <= elem) &&
                (forall|w: int| f < w < v@.len() ==> v@[w] > elem)
            decreases f + 1 - c
        {
            let m: usize = c + (f - c) / 2;
            if v[m] <= elem {
                c = m + 1;
            } else {
                if m == 0 {
                    return -1;
                }
                f = m - 1;
            }
        }
        
        // At this point, c > f, so return c - 1
        if c == 0 {
            -1
        } else {
            (c - 1) as i32
        }
    }

    fn search(v: &[i32], elem: i32) -> (b: bool)
        requires sorted(as_int_seq(v@))
        ensures b == as_int_seq(v@).contains(elem as int)
    {
        let p = binary_search(v, elem);
        if p == -1 {
            false
        } else {
            v[p as usize] == elem
        }
    }

    fn binary_search_rec(v: &[i32], elem: i32, c: usize, f: usize) -> (p: i32)
        requires 
            sorted(as_int_seq(v@)) &&
            c <= f + 1 &&
            f < v@.len() &&
            (forall|k: int| 0 <= k < c ==> v@[k] <= elem) &&
            (forall|k: int| f < k < v@.len() ==> v@[k] > elem)
        decreases f + 1 - c
        ensures 
            -1 <= p < v@.len() &&
            (forall|u: int| 0 <= u <= p ==> v@[u] <= elem) &&
            (forall|w: int| p < w < v@.len() ==> v@[w] > elem)
    {
        if c == f + 1 {
            if c == 0 { -1 } else { (c - 1) as i32 }
        } else {
            let m: usize = c + (f - c) / 2;
            if v[m] <= elem {
                binary_search_rec(v, elem, m + 1, f)
            } else {
                if m == 0 {
                    -1
                } else {
                    binary_search_rec(v, elem, c, m - 1)
                }
            }
        }
    }

    fn other_b_search(v: &[i32], elem: i32) -> (result: (bool, usize))
        requires sorted(as_int_seq(v@))
        ensures ({
            let (b, p) = result;
            p <= v@.len() &&
            b == as_int_seq(v@).contains(elem as int) &&
            (b ==> p < v@.len() && v@[p] == elem) &&
            (!b ==> (forall|u: int| 0 <= u < p ==> v@[u] < elem) &&
                    (forall|w: int| p <= w < v@.len() ==> v@[w] > elem))
        })
    {
        let p_signed = binary_search(v, elem);
        let b: bool;
        let p: usize;
        
        if p_signed == -1 {
            b = false;
            p = 0;
        } else {
            let p_unsigned = p_signed as usize;
            b = v[p_unsigned] == elem;
            p = if b { p_unsigned } else { p_unsigned + 1 };
        }
        
        (b, p)
    }
}

fn main() {}