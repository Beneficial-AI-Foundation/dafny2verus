use vstd::prelude::*;

verus! {
    // Algorithm 1: From left to right return the first
    fn mmaximum1(v: &[int]) -> (i: usize)
        requires v.len() > 0,
        ensures 
            i < v.len(),
            forall |k: int| 0 <= k < v.len() ==> v[i as int] >= v[k],
    {
        let mut j: usize = 1;
        let mut i: usize = 0;
        while j < v.len()
            invariant
                0 <= j <= v.len(),
                i < j,
                forall |k: int| 0 <= k < j ==> v[i as int] >= v[k],
            decreases v.len() - j,
        {
            if v[j] > v[i] {
                i = j;
            }
            j = j + 1;
        }
        i
    }

    // Algorithm 2: From right to left return the last
    fn mmaximum2(v: &[int]) -> (i: usize)
        requires v.len() > 0,
        ensures 
            i < v.len(),
            forall |k: int| 0 <= k < v.len() ==> v[i as int] >= v[k],
    {
        let mut j: usize = v.len() - 1;
        let mut i: usize = v.len() - 1;
        while j > 0
            invariant
                i < v.len(),
                j <= v.len() - 1,
                forall |k: int| j <= k < v.len() ==> v[k] <= v[i as int],
            decreases j,
        {
            j = j - 1;
            if v[j] > v[i] {
                i = j;
            }
        }
        i
    }

    fn mfirstMaximum(v: &[int]) -> (i: usize)
        requires v.len() > 0,
        ensures 
            i < v.len(),
            forall |k: int| 0 <= k < v.len() ==> v[i as int] >= v[k],
            forall |l: int| 0 <= l < i ==> v[i as int] > v[l],
    {
        let mut j: usize = 1;
        let mut i: usize = 0;
        while j < v.len()
            invariant
                0 <= j <= v.len(),
                i < j,
                forall |k: int| 0 <= k < j ==> v[i as int] >= v[k],
                forall |k: int| 0 <= k < i ==> v[i as int] > v[k],
            decreases v.len() - j,
        {
            if v[j] > v[i] {
                i = j;
            }
            j = j + 1;
        }
        i
    }

    fn mlastMaximum(v: &[int]) -> (i: usize)
        requires v.len() > 0,
        ensures 
            i < v.len(),
            forall |k: int| 0 <= k < v.len() ==> v[i as int] >= v[k],
            forall |l: int| i < l < v.len() ==> v[i as int] > v[l],
    {
        let mut j: usize = v.len() - 1;
        let mut i: usize = v.len() - 1;
        while j > 0
            invariant
                i < v.len(),
                j <= v.len() - 1,
                forall |k: int| j <= k < v.len() ==> v[k] <= v[i as int],
                forall |k: int| i < k < v.len() ==> v[k] < v[i as int],
            decreases j,
        {
            j = j - 1;
            if v[j] > v[i] {
                i = j;
            }
        }
        i
    }

    // Algorithm: from left to right
    fn mmaxvalue1(v: &[int]) -> (m: int)
        requires v.len() > 0,
        ensures 
            exists |k: int| 0 <= k < v.len() && m == v[k],
            forall |k: int| 0 <= k < v.len() ==> m >= v[k],
    {
        let i = mmaximum1(v);
        v[i]
    }

    // Algorithm: from right to left
    fn mmaxvalue2(v: &[int]) -> (m: int)
        requires v.len() > 0,
        ensures 
            exists |k: int| 0 <= k < v.len() && m == v[k],
            forall |k: int| 0 <= k < v.len() ==> m >= v[k],
    {
        let i = mmaximum2(v);
        v[i]
    }
}