use vstd::prelude::*;

verus! {
    // Predicate to check if a segment of array is sorted
    spec fn sorted_seg(a: Seq<int>, i: int, j: int) -> bool 
        recommends 0 <= i <= j <= a.len()
    {
        forall|l: int, k: int| i <= l <= k < j ==> a[l] <= a[k]
    }

    fn sel_sort(a: &mut Vec<int>, c: usize, f: usize)
        requires 
            c <= f,
            f <= old(a).len(),
        ensures 
            a.len() == old(a).len(),
            sorted_seg(a@, c as int, f as int),
    {
        if c < f { // at least one element
            let mut i = c;
            while i + 1 < f
                invariant
                    c <= i < f,
                    i < a.len(),
                    f <= a.len(),
                    a.len() == old(a).len(),
                    sorted_seg(a@, c as int, i as int),
                    forall|k: int, l: int| c as int <= k < i as int && i as int <= l < f as int ==> a@[k] <= a@[l],
                decreases f - i
            {
                // Find the minimum element in [i, f)
                let mut less = i;
                let mut j = i + 1;
                while j < f
                    invariant
                        i + 1 <= j <= f,
                        i <= less < f,
                        less < a.len(),
                        j <= a.len(),
                        f <= a.len(),
                        a.len() == old(a).len(),
                        forall|k: int| i as int <= k < j as int ==> a@[less as int] <= a@[k],
                    decreases f - j
                {
                    if j < a.len() && less < a.len() && a[j] < a[less] {
                        less = j;
                    }
                    j = j + 1;
                }
                // Swap elements a[i] and a[less]
                if i < a.len() && less < a.len() {
                    let temp_i = a[i];
                    let temp_less = a[less];
                    a.set(i, temp_less);
                    a.set(less, temp_i);
                }
                i = i + 1;
            }
        }
    }
}

fn main() {}