use vstd::prelude::*;

verus! {
    fn concat(a: &[i32], b: &[i32]) -> (c: Vec<i32>)
        requires
            a.len() + b.len() < usize::MAX
        ensures
            c.len() == b.len() + a.len(),
            forall|k: int| 0 <= k < a.len() ==> c[k as int] == a[k as int],
            forall|k: int| 0 <= k < b.len() ==> c[(k + a.len()) as int] == b[k as int]
    {
        let total_len = a.len() + b.len();
        let mut c = Vec::with_capacity(total_len);
        let mut i: usize = 0;
        
        while i < total_len
            invariant
                0 <= i <= total_len,
                total_len == a.len() + b.len(),
                c.len() == i,
                forall|k: int| 0 <= k < i && k < a.len() ==> c[k as int] == a[k as int],
                forall|k: int| 0 <= k < i && k >= a.len() ==> {
                    &&& k - a.len() >= 0
                    &&& k - a.len() < b.len()
                    &&& c[k as int] == b[(k - a.len()) as int]
                }
            decreases total_len - i
        {
            if i < a.len() {
                c.push(a[i]);
            } else {
                c.push(b[i - a.len()]);
            }
            i = i + 1;
        }
        
        c
    }
}