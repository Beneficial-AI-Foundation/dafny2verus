use vstd::prelude::*;

verus! {
    // Predicate checks if elements of a are in ascending order, two additional conditions are added to allow us to sort in specific range within array
    spec fn sorted(a: Seq<int>, from: int, to: int) -> bool
        recommends 0 <= from <= to <= a.len()
    {
        forall|x: int, y: int| from <= x < y < to ==> a[x] <= a[y]
    }

    // Helps ensure swapping is valid, it is used inside the nested while loop to make sure linear order is being kept 
    spec fn pivot(a: Seq<int>, to: int, pvt: int) -> bool
        recommends 0 <= pvt < to <= a.len()
    {
        forall|x: int, y: int| 0 <= x < pvt < y < to ==> a[x] <= a[y]
    }

    // Here having the algorithm for the bubblesort
    fn bubble_sort(a: &mut Vec<int>)
        requires 
            old(a).len() > 0,
        ensures 
            sorted(a@, 0, a.len() as int),
            a.len() == old(a).len(),
    {
        let mut i = 1;

        while i < a.len()
            invariant 
                i <= a.len(),
                sorted(a@, 0, i as int),
                a.len() == old(a).len(),
            decreases a.len() - i
        {
            let mut j = i;

            while j > 0
                invariant 
                    a.len() == old(a).len(),
                    j <= i,
                    i < a.len(),
                    0 <= j <= i + 1 <= a.len(),
                    sorted(a@, 0, j as int),
                    sorted(a@, j as int, (i + 1) as int),
                    pivot(a@, (i + 1) as int, j as int),
                decreases j
            {
                if j > 0 && j - 1 < a.len() && j < a.len() && a[j - 1] > a[j] {
                    // Swap elements
                    let temp = a[j - 1];
                    let val_j = a[j];
                    a.set(j - 1, val_j);
                    a.set(j, temp);
                }
                j = j - 1;
            }
            i = i + 1;
        }
    }

    fn main() {
    }
}