use vstd::prelude::*;

verus! {
    // Author: Snorri Agnarsson, snorri@hi.is
    
    // Search1000 is a Verus version of a function shown
    // by Jon Bentley in his old Programming Pearls
    // column in CACM.  Surprisingly Verus needs no help
    // to verify the function.
    fn search1000(a: &[int], x: int) -> (k: usize)
        requires 
            a.len() >= 1000,
            forall|p: int, q: int| 0 <= p < q < 1000 ==> a[p] <= a[q],
        ensures 
            0 <= k <= 1000,
            forall|r: int| 0 <= r < k ==> a[r] < x,
            forall|r: int| k <= r < 1000 ==> a[r] >= x,
    {
        let mut k: usize = 0;
        if a[500] < x { k = 489; }
        if a[k+255] < x { k = k+256; }
        if a[k+127] < x { k = k+128; }
        if a[k+63] < x  { k = k+64;  }
        if a[k+31] < x  { k = k+32;  }
        if a[k+15] < x  { k = k+16;  }
        if a[k+7] < x   { k = k+8;   }
        if a[k+3] < x   { k = k+4;   }
        if a[k+1] < x   { k = k+2;   }
        if a[k] < x     { k = k+1;   }
        k
    }
    
    // Is2Pow(n) is true iff n==2^k for some k>=0.
    spec fn is_2_pow(n: int) -> bool
        decreases n
    {
        if n < 1 {
            false
        } else if n == 1 {
            true
        } else {
            n % 2 == 0 && is_2_pow(n / 2)
        }
    }
    
    // This method is a binary search that only works for array
    // segments of size n == 2^k-1 for some k>=0.
    fn search_2_pow_loop(a: &[int], i: usize, n: usize, x: int) -> (k: usize)
        requires 
            0 <= i && i + n <= a.len(),
            forall|p: int, q: int| i <= p < q < i + n ==> a[p] <= a[q],
            is_2_pow((n + 1) as int),
        ensures 
            i <= k && k <= i + n,
            forall|r: int| i <= r < k ==> a[r] < x,
            forall|r: int| k <= r < i + n ==> a[r] >= x,
    {
        let mut k = i;
        let mut c = n;
        while c != 0
            decreases c
        {
            c = c / 2;
            if a[k + c] < x { k = k + c + 1; }
        }
        k
    }
    
    // This method is a binary search that only works for array
    // segments of size n == 2^k-1 for some k>=0.
    fn search_2_pow_recursive(a: &[int], i: usize, n: usize, x: int) -> (k: usize)
        requires 
            0 <= i && i + n <= a.len(),
            forall|p: int, q: int| i <= p < q < i + n ==> a[p] <= a[q],
            is_2_pow((n + 1) as int),
        ensures 
            i <= k && k <= i + n,
            forall|r: int| i <= r < k ==> a[r] < x,
            forall|r: int| k <= r < i + n ==> a[r] >= x,
        decreases n
    {
        if n == 0 { return i; }
        if a[i + n / 2] < x {
            search_2_pow_recursive(a, i + n / 2 + 1, n / 2, x)
        } else {
            search_2_pow_recursive(a, i, n / 2, x)
        }
    }
}

fn main() {}