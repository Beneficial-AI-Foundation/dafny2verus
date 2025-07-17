use vstd::prelude::*;

verus! {
    fn increment_matrix(a: &mut Vec<Vec<i32>>)
        requires 
            old(a).len() > 0,
            forall|i: int| 0 <= i < old(a).len() ==> 
                #[trigger] old(a)[i].len() == old(a)[0].len(),
        ensures
            a.len() == old(a).len(),
            forall|i: int| 0 <= i < a.len() ==> 
                #[trigger] a[i].len() == old(a)[i].len(),
            forall|i: int, j: int| 0 <= i < a.len() && 0 <= j < a[0].len() ==> 
                #[trigger] a[i][j] == old(a)[i][j] + 1,
    {
        let mut m: usize = 0;
        let len0 = a.len();
        let len1 = if len0 > 0 { a[0].len() } else { 0 };
        
        while m != len0
            invariant
                0 <= m <= len0,
                len0 == a.len(),
                forall|i: int| 0 <= i < a.len() ==> 
                    #[trigger] a[i].len() == len1,
                forall|i: int, j: int| 0 <= i < m && 0 <= j < len1 ==> 
                    #[trigger] a[i][j] == old(a)[i][j] + 1,
                forall|i: int, j: int| m <= i < len0 && 0 <= j < len1 ==> 
                    #[trigger] a[i][j] == old(a)[i][j],
            decreases len0 - m,
        {
            let mut n: usize = 0;
            
            while n != len1
                invariant
                    0 <= n <= len1,
                    len0 == a.len(),
                    forall|i: int| 0 <= i < a.len() ==> 
                        #[trigger] a[i].len() == len1,
                    forall|i: int, j: int| 0 <= i < m && 0 <= j < len1 ==> 
                        #[trigger] a[i][j] == old(a)[i][j] + 1,
                    forall|i: int, j: int| m < i < len0 && 0 <= j < len1 ==> 
                        #[trigger] a[i][j] == old(a)[i][j],
                    forall|j: int| 0 <= j < n ==> 
                        #[trigger] a[m as int][j] == old(a)[m as int][j] + 1,
                    forall|j: int| n <= j < len1 ==> 
                        #[trigger] a[m as int][j] == old(a)[m as int][j],
                decreases len1 - n,
            {
                let old_val = a[m][n];
                let mut new_row = a[m].clone();
                new_row.set(n, old_val + 1);
                a.set(m, new_row);
                n = n + 1;
            }
            m = m + 1;
        }
    }
}