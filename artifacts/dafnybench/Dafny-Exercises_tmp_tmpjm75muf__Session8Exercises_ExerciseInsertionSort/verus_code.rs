use vstd::prelude::*;

verus! {
    spec fn sorted_seg(a: Seq<int>, i: int, j: int) -> bool
        recommends 0 <= i <= j + 1 <= a.len()
    {
        forall|l: int, k: int| i <= l <= k <= j ==> a[l] <= a[k]
    }

    fn insertion_sort(a: &mut Vec<int>)
        requires old(a).len() > 0,
        ensures 
            sorted_seg(a@, 0, a.len() as int - 1),
            a@.to_multiset() == old(a)@.to_multiset(),
    {
        let mut i: usize = 0;
        
        while i < a.len()
            invariant 
                0 <= i <= a.len(),
                i == 0 ==> sorted_seg(a@, 0, -1),
                i > 0 ==> sorted_seg(a@, 0, i as int - 1),
                a@.to_multiset() == old(a)@.to_multiset(),
            decreases a.len() - i,
        {
            let temp = a[i];
            let mut j = i;
            
            while j > 0 && temp < a[j - 1]
                invariant 
                    0 <= j <= i,
                    j < a.len(),
                decreases j,
            {
                let prev_val = a[j - 1];
                a.set(j, prev_val);
                j = j - 1;
            }
            
            a.set(j, temp);
            i = i + 1;
        }
    }
}

fn main() {}