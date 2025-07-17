use vstd::prelude::*;

verus! {
    fn max(a: &[int], b: &[int], i: usize, j: usize) -> (m: int)
        requires 
            0 <= i < a.len(),
            0 <= j < b.len(),
        ensures
            a[i as int] > b[j as int] ==> m == a[i as int],
            a[i as int] <= b[j as int] ==> m == b[j as int],
    {
        if a[i] > b[j] {
            a[i]
        } else {
            b[j]
        }
    }

    fn testMax(a: &[int], b: &[int], i: usize, j: usize)
        requires 
            0 <= i < a.len(),
            0 <= j < b.len(),
    {
        let max_val = max(a, b, i, j);
        assert(a[i as int] > b[j as int] ==> max_val == a[i as int]);
        assert(a[i as int] <= b[j as int] ==> max_val == b[j as int]);
    }
}

fn main() {}