use vstd::prelude::*;

verus! {
    spec fn sorted(a: &[int]) -> bool {
        forall|i: int, j: int| 0 <= i < j < a.len() ==> a[i] <= a[j]
    }

    fn binary_search(a: &[int], x: int) -> (index: i32)
        requires sorted(a),
        ensures 
            (0 <= index < a.len()) ==> a[index as int] == x,
            (index == -1) ==> forall|i: int| 0 <= i < a.len() ==> a[i] != x,
    {
        let mut i = 0usize;
        while i < a.len()
            invariant 
                0 <= i <= a.len(),
                forall|j: int| 0 <= j < i ==> a[j] != x,
            decreases a.len() - i,
        {
            if a[i] == x {
                proof {
                    assert(a[i as int] == x);
                    assert(0 <= i < a.len());
                }
                return i as i32;
            }
            i += 1;
        }
        proof {
            assert(i == a.len());
            assert(forall|j: int| 0 <= j < i ==> a[j] != x);
            assert(forall|j: int| 0 <= j < a.len() ==> a[j] != x);
        }
        -1
    }
}

fn main() {}