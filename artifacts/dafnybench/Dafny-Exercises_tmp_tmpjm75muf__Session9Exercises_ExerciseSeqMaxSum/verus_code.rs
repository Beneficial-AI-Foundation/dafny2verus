use vstd::prelude::*;

verus! {
    // Function to compute sum of array elements from index i to j-1
    spec fn sum(v: &[int], i: int, j: int) -> int
        recommends 0 <= i <= j <= v.len()
        decreases j - i
    {
        if i >= j {
            0
        } else {
            sum(v, i, j - 1) + v[j - 1]
        }
    }

    // Lemma to prove sum equivalence
    proof fn sum_single(v: &[int], i: int)
        requires 0 <= i < v.len()
        ensures sum(v, i, i + 1) == v[i]
    {
        assert(sum(v, i, i + 1) == sum(v, i, i) + v[i]);
        assert(sum(v, i, i) == 0);
    }

    // Method to find maximum sum subarray ending at position i
    fn seg_max_sum(v: &[int], i: usize) -> (result: (int, usize))
        requires 
            v.len() > 0,
            i < v.len(),
        ensures 
            result.1 <= i,
            result.0 == sum(v, result.1 as int, i as int + 1),
    {
        let mut s = v[0];
        let mut k = 0;
        let mut j = 0;
        
        proof {
            sum_single(v, 0);
        }
        
        while j < i
            invariant
                0 <= j <= i,
                0 <= k <= j,
                j < v.len(),
                i < v.len(),
                s == sum(v, k as int, j as int + 1),
            decreases i - j
        {
            if s + v[j + 1] > v[j + 1] {
                s = s + v[j + 1];
                proof {
                    assert(s == sum(v, k as int, j as int + 1) + v[j + 1]);
                    assert(s == sum(v, k as int, j as int + 2));
                }
            } else {
                k = j + 1;
                s = v[j + 1];
                proof {
                    sum_single(v, j as int + 1);
                    assert(s == sum(v, j as int + 1, j as int + 2));
                }
            }
            j = j + 1;
        }
        
        (s, k)
    }

    // Alternative sum function (left to right)
    spec fn sum2(v: &[int], i: int, j: int) -> int
        recommends 0 <= i <= j <= v.len()
        decreases j - i
    {
        if i >= j {
            0
        } else {
            v[i] + sum2(v, i + 1, j)
        }
    }

    // Lemma for sum2
    proof fn sum2_single(v: &[int], i: int)
        requires 0 <= i < v.len()
        ensures sum2(v, i, i + 1) == v[i]
    {
        assert(sum2(v, i, i + 1) == v[i] + sum2(v, i + 1, i + 1));
        assert(sum2(v, i + 1, i + 1) == 0);
    }

    // Method to find maximum sum subarray ending at position i (right to left)
    fn seg_suma_maxima2(v: &[int], i: usize) -> (result: (int, usize))
        requires 
            v.len() > 0,
            i < v.len(),
        ensures 
            result.1 <= i,
            result.0 == sum2(v, result.1 as int, i as int + 1),
    {
        let mut s = v[i];
        let mut k = i;
        let mut j = i;
        let mut maxs = s;
        
        proof {
            sum2_single(v, i as int);
        }
        
        while j > 0
            invariant
                0 <= j <= i,
                0 <= k <= i,
                i < v.len(),
                s == sum2(v, j as int, i as int + 1),
                maxs == sum2(v, k as int, i as int + 1),
            decreases j
        {
            s = s + v[j - 1];
            if s > maxs {
                maxs = s;
                k = j - 1;
            }
            j = j - 1;
        }
        
        s = maxs;
        (s, k)
    }
}

fn main() {}