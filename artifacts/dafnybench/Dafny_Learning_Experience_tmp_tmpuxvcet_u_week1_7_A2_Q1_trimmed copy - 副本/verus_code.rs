use vstd::prelude::*;

verus! {
    // Ghost function to count even numbers in sequence up to index hi
    spec fn count(hi: nat, s: Seq<int>) -> int
        decreases hi
    {
        if hi == 0 {
            0
        } else if s[hi - 1] % 2 == 0 {
            1 + count((hi - 1) as nat, s)
        } else {
            count((hi - 1) as nat, s)
        }
    }

    // Method to compute count with memoization in array b
    fn foo_count(count_index: usize, a: Seq<int>, b: &mut Vec<i32>) -> (p: usize) {
        requires(count_index == 0 || (a.len() == b.len() && 1 <= count_index <= a.len()));
        decreases(count_index);
        ensures(|p: usize| proof { p == count(count_index as nat, a) as usize });
        
        if count_index == 0 {
            0
        } else {
            if a.index(count_index - 1) % 2 == 0 {
                let d = foo_count(count_index - 1, a, b);
                let p = d + 1;
                b.set(count_index - 1, p as i32);
                p
            } else {
                let d = foo_count(count_index - 1, a, b);
                b.set(count_index - 1, d as i32);
                d
            }
        }
    }

    // Precompute all count values
    fn foo_precompute(a: Vec<i32>, b: &mut Vec<i32>) {
        requires(a.len() == b.len());
        
        let mut count_index = 1;
        while count_index != a.len() + 1 {
            invariant(1 <= count_index <= a.len() + 1);
            decreases(a.len() + 1 - count_index);
            
            let _p = foo_count(count_index, a@, b);
            count_index = count_index + 1;
        }
    }

    // Simplified compute count method
    fn compute_count(count_index: usize, a: Seq<int>, b: &mut Vec<i32>) -> (p: usize) {
        requires(count_index == 0 || (a.len() == b.len() && 1 <= count_index <= a.len()));
        decreases(count_index);
        ensures(|p: usize| proof { p == count(count_index as nat, a) as usize });
        
        if count_index == 0 {
            0
        } else {
            if a.index(count_index - 1) % 2 == 0 {
                let d = compute_count(count_index - 1, a, b);
                let p = d + 1;
                b.set(count_index - 1, p as i32);
                p
            } else {
                let d = compute_count(count_index - 1, a, b);
                b.set(count_index - 1, d as i32);
                d
            }
        }
    }

    // Precompute wrapper
    fn precompute(a: Vec<i32>, b: &mut Vec<i32>) -> (p: usize) {
        requires(a.len() == b.len());
        ensures(|p: usize| (b.len() == 0 || (a.len() == b.len() && 1 <= b.len() <= a.len())));
        
        compute_count(b.len(), a@, b)
    }

    // Main evens method - simplified version
    fn evens(a: Vec<i32>) -> Vec<Vec<i32>> {
        let mut b = Vec::new();
        b.resize(a.len(), 0);
        
        let _foo = precompute(a, &mut b);
        
        let mut c = Vec::new();
        c.resize(a.len(), Vec::new());
        
        let mut m = 0;
        while m != a.len() {
            invariant(0 <= m <= a.len());
            decreases(a.len() - m);
            
            c.set(m, Vec::new());
            c[m].resize(a.len(), 0);
            
            let mut n = 0;
            while n != a.len() {
                invariant(0 <= n <= a.len());
                decreases(a.len() - n);
                
                if n < m {
                    c[m].set(n, 0);
                } else {
                    if m > 0 {
                        c[m].set(n, b[n] - b[m - 1]);
                    } else {
                        c[m].set(n, b[n]);
                    }
                }
                n = n + 1;
            }
            m = m + 1;
        }
        c
    }

    // Multiplication method
    fn mult(x: i32, y: i32) -> (r: i32) {
        requires(x >= 0 && y >= 0);
        decreases(x);
        ensures(|r: i32| x >= 0 && y >= 0 ==> r == x * y);
        
        if x == 0 {
            0
        } else {
            let z = mult(x - 1, y);
            z + y
        }
    }
}

fn main() {}