use vstd::prelude::*;

verus! {
    // First method: LinearSearch - for integer arrays
    fn linear_search_int(a: &[i32], target: i32) -> (n: i32)
        requires true,
        ensures
            -1 <= n < a.len(),
            n == -1 || a[n as int] == target,
            n != -1 ==> forall|i: int| 0 <= i < n ==> a[i] != target,
            n == -1 ==> forall|i: int| 0 <= i < a.len() ==> a[i] != target
    {
        let mut n: usize = 0;
        
        while n < a.len()
            invariant
                0 <= n <= a.len(),
                forall|i: int| 0 <= i < n ==> a[i] != target,
            decreases a.len() - n
        {
            if a[n] == target {
                return n as i32;
            }
            n = n + 1;
        }
        -1
    }

    // Second method: LinearSearch1 - search with length constraint
    fn linear_search1_int(a: &[i32], target: i32, s1_len: usize) -> (n: i32)
        requires 
            s1_len <= a.len(),
        ensures
            -1 <= n < a.len(),
            n == -1 || a[n as int] == target,
            n != -1 ==> forall|i: int| 0 <= i < n ==> a[i] != target,
            n == -1 ==> forall|i: int| 0 <= i < s1_len ==> a[i] != target
    {
        let mut n: usize = 0;
        
        while n < s1_len
            invariant
                0 <= n <= s1_len,
                forall|i: int| 0 <= i < n ==> a[i] != target,
            decreases s1_len - n
        {
            if a[n] == target {
                return n as i32;
            }
            n = n + 1;
        }
        -1
    }

    // Third method: LinearSearch2 - reverse search
    fn linear_search2_int(data: &[i32], element: i32, s1_len: usize) -> (position: i32)
        requires 
            s1_len <= data.len(),
        ensures
            position == -1 || position >= 1,
            position >= 1 ==> exists|i: int| 0 <= i < s1_len && data[i] == element,
            position == -1 ==> forall|i: int| 0 <= i < s1_len ==> data[i] != element
    {
        let mut n: usize = 0;
        let mut position: i32 = 0;
        
        while n < s1_len
            invariant
                0 <= n <= s1_len,
                position >= 1 ==> exists|i: int| 0 <= i < s1_len && data[i] == element,
                forall|i: int| s1_len - 1 - n < i < s1_len ==> data[i] != element,
            decreases s1_len - n
        {
            let index = s1_len - 1 - n;
            if data[index] == element {
                position = n as i32 + 1;
                return position;
            }
            n = n + 1;
        }
        -1
    }

    // Fourth method: LinearSearch3 - reverse indexed search
    fn linear_search3_int(data: &[i32], element: i32, s1_len: usize) -> (position: i32)
        requires 
            s1_len <= data.len(),
        ensures
            position == -1 || position >= 1,
            position >= 1 ==> s1_len != 0,
    {
        let mut n: usize = 0;
        let n1: usize = s1_len;
        let mut position: i32 = 0;
        
        while n < s1_len
            invariant
                0 <= n <= s1_len,
                position >= 1 ==> s1_len != 0,
                forall|i: int| data.len() - n1 < i < data.len() - n1 + n ==> data[i] != element,
            decreases s1_len - n
        {
            let index = data.len() - n1 + n;
            if data[index] == element {
                position = n as i32 + 1;
                return position;
            }
            n = n + 1;
        }
        
        -1
    }

    fn main() {
    }
}