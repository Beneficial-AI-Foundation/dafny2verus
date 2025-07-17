use vstd::prelude::*;

verus! {
    // First version of linear search - finds first element equal to target
    fn linear_search0(a: &Vec<u32>, target: u32) -> (n: usize)
        requires true,
        ensures 
            n <= a.len(),
            n < a.len() ==> a[n as int] == target,
    {
        let mut n = 0;
        while n < a.len()
            invariant 
                n <= a.len(),
            decreases a.len() - n,
        {
            if a[n] == target {
                return n;
            }
            n = n + 1;
        }
        n
    }

    // Enhanced version with stronger postcondition  
    fn linear_search1(a: &Vec<u32>, target: u32) -> (n: usize)
        requires true,
        ensures 
            n <= a.len(),
            n < a.len() ==> a[n as int] == target,
            n == a.len() ==> (forall|i: int| 0 <= i < a.len() ==> a[i] != target),
    {
        let mut n = 0;
        while n < a.len()
            invariant 
                n <= a.len(),
                forall|i: int| 0 <= i < n ==> a[i] != target,
            decreases a.len() - n,
        {
            if a[n] == target {
                return n;
            }
            n = n + 1;
        }
        n
    }

    // Test function
    fn test_linear_search() {
        let a: Vec<u32> = vec![1, 2, 3];
        let n = linear_search1(&a, 2);
        assert(n == 1 || n == 2 || n == 3 || n == 0);
    }

    fn main() {
        test_linear_search();
    }
}