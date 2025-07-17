use vstd::prelude::*;

verus! {
    fn allDigits(s: Vec<char>) -> (result: bool)
        ensures result <==> (forall|i: int| 0 <= i < s.len() ==> {
            let c = #[trigger] s[i];
            c >= '0' && c <= '9'
        })
    {
        let mut result = true;
        let mut i: usize = 0;
        
        while i < s.len()
            invariant 
                result <==> (forall|ii: int| 0 <= ii < i ==> {
                    let c = #[trigger] s[ii];
                    c >= '0' && c <= '9'
                }),
                i <= s.len()
            decreases s.len() - i
        {
            let c = s[i];
            if !(c >= '0' && c <= '9') {
                return false;
            }
            i += 1;
        }
        
        result
    }
}

fn main() {}