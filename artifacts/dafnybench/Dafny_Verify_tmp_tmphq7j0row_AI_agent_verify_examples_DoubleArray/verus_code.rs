use vstd::prelude::*;

verus! {
    fn double_array(src: &Vec<u32>, dst: &mut Vec<u32>)
        requires 
            src.len() == old(dst).len(),
            forall|i: int| 0 <= i < src.len() ==> src[i] <= u32::MAX / 2,
        ensures
            dst.len() == old(dst).len(),
            forall|i: int| 0 <= i < src.len() ==> dst[i] == 2 * src[i],
    {
        let mut n: usize = 0;
        while n < src.len()
            invariant
                0 <= n <= src.len(),
                dst.len() == old(dst).len(),
                forall|i: int| 0 <= i < src.len() ==> src[i] <= u32::MAX / 2,
                forall|i: int| 0 <= i < n ==> dst[i] == 2 * src[i],
            decreases src.len() - n,
        {
            dst[n] = 2 * src[n];
            n = n + 1;
        }
    }
}