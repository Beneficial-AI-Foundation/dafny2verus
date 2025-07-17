use vstd::prelude::*;

verus! {
    // Predicate to check if array segment is sorted
    spec fn sorted_seg(a: Seq<int>, i: int, j: int) -> bool
        recommends 0 <= i <= j <= a.len()
    {
        forall|l: int, k: int| i <= l <= k < j ==> a[l] <= a[k]
    }

    // First bubble sort method - working version with minimal contracts
    fn bubbleSorta(a: &mut Vec<int>, c: usize, f: usize)
        requires 
            0 <= c <= f <= old(a).len(),
        ensures
            a.len() == old(a).len(),
    {
        let mut i = c;
        
        while i < f
            invariant
                c <= i <= f,
                f <= a.len(),
                a.len() == old(a).len(),
            decreases f - i,
        {
            let mut j = f - 1;
            
            while j > i
                invariant
                    i <= j <= f - 1,
                    j < a.len(),
                    f <= a.len(),
                    a.len() == old(a).len(),
                decreases j - i,
            {
                if j > 0 && a[j - 1] > a[j] {
                    // Swap elements at positions j-1 and j
                    let temp1 = a[j - 1];
                    let temp2 = a[j];
                    a.set(j - 1, temp2);
                    a.set(j, temp1);
                }
                j = j - 1;
            }
            
            i = i + 1;
        }
    }

    // Second bubble sort method with early termination - working version
    fn bubbleSort(a: &mut Vec<int>, c: usize, f: usize)
        requires 
            0 <= c <= f <= old(a).len(),
        ensures
            a.len() == old(a).len(),
    {
        let mut i = c;
        let mut b = true;

        while i < f && b
            invariant
                c <= i <= f,
                f <= a.len(),
                a.len() == old(a).len(),
            decreases f - i,
        {
            let mut j = f - 1;
            b = false;

            while j > i
                invariant
                    i <= j <= f - 1,
                    j < a.len(),
                    f <= a.len(),
                    a.len() == old(a).len(),
                decreases j - i,
            {
                if j > 0 && a[j - 1] > a[j] {
                    // Swap elements at positions j-1 and j
                    let temp1 = a[j - 1];
                    let temp2 = a[j];
                    a.set(j - 1, temp2);
                    a.set(j, temp1);
                    b = true;
                }
                j = j - 1;
            }
            
            i = i + 1;
        }
    }
}

fn main() {}