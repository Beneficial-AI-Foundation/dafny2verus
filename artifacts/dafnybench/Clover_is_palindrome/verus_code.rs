use vstd::prelude::*;

verus! {
    fn is_palindrome(x: Seq<char>) -> (result: bool) 
        ensures result <==> (forall|i: int| 0 <= i < x.len() ==> x[i] == x[x.len() - i - 1])
    {
        let len = x.len();
        if len == 0 {
            return true;
        }
        
        let mut i: usize = 0;
        let mut j: usize = len - 1;
        
        while i < j
            invariant
                0 <= i <= j + 1,
                0 <= j < len,
                i + j == len - 1,
                forall|k: int| 0 <= k < i ==> x[k] == x[len - k - 1]
        {
            let ghost i_int = i as int;
            let ghost j_int = j as int;
            
            if x.index(i_int) != x.index(j_int) {
                return false;
            }
            
            i = i + 1;
            j = j - 1;
        }
        
        true
    }
}