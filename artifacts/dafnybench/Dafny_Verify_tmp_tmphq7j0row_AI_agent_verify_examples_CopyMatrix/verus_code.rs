use vstd::prelude::*;

verus! {
    // Matrix copy function - translated from Dafny
    fn copy_matrix(src: Vec<Vec<i32>>, dst: &mut Vec<Vec<i32>>)
        requires 
            src.len() == old(dst).len(),
            src.len() > 0,
            old(dst).len() > 0,
            forall|i: int| 0 <= i < src.len() ==> #[trigger] src[i].len() == src[0].len(),
            forall|i: int| 0 <= i < old(dst).len() ==> #[trigger] old(dst)[i].len() == old(dst)[0].len(),
            src[0].len() == old(dst)[0].len(),
        ensures
            dst.len() == src.len(),
            forall|i: int| 0 <= i < dst.len() ==> #[trigger] dst[i].len() == dst[0].len(),
            forall|i: int, j: int| 0 <= i < src.len() && 0 <= j < src[0].len() ==> 
                dst[i][j] == src[i][j],
    {
        let mut m: usize = 0;
        while m != src.len()
            invariant
                0 <= m <= src.len(),
                src.len() == dst.len(),
                src.len() > 0,
                dst.len() > 0,
                forall|i: int| 0 <= i < src.len() ==> #[trigger] src[i].len() == src[0].len(),
                forall|i: int| 0 <= i < dst.len() ==> #[trigger] dst[i].len() == dst[0].len(),
                src[0].len() == dst[0].len(),
                forall|i: int, j: int| 0 <= i < m && 0 <= j < src[0].len() ==> 
                    dst[i][j] == src[i][j],
            decreases src.len() - m,
        {
            let mut n: usize = 0;
            while n != src[m].len()
                invariant
                    0 <= n <= src[m as int].len(),
                    src.len() == dst.len(),
                    src.len() > 0,
                    dst.len() > 0,
                    m < src.len(),
                    forall|i: int| 0 <= i < src.len() ==> #[trigger] src[i].len() == src[0].len(),
                    forall|i: int| 0 <= i < dst.len() ==> #[trigger] dst[i].len() == dst[0].len(),
                    src[0].len() == dst[0].len(),
                    src[m as int].len() == dst[m as int].len(),
                    forall|i: int, j: int| 0 <= i < m && 0 <= j < src[0].len() ==> 
                        dst[i][j] == src[i][j],
                    forall|j: int| 0 <= j < n ==> 
                        dst[m as int][j] == src[m as int][j],
                decreases src[m as int].len() - n,
            {
                // Copy the value from source to destination
                let val = src[m][n];
                let mut inner_row = dst[m].clone();
                inner_row.set(n, val);
                dst.set(m, inner_row);
                n = n + 1;
            }
            m = m + 1;
        }
    }

    fn main() {}
}