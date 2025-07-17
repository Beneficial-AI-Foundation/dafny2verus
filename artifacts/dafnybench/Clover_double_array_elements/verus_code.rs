use vstd::prelude::*;

verus! {
    fn double_array_elements(s: &mut Vec<i32>)
        requires forall|i: int| 0 <= i < old(s).len() ==> #[trigger] old(s)[i] >= -1073741824 && old(s)[i] <= 1073741823, // prevent overflow
        ensures 
            forall|i: int| 0 <= i < old(s).len() ==> #[trigger] s[i] == 2 * old(s)[i],
            s.len() == old(s).len(),
    {
        let mut i: usize = 0;
        let ghost old_s = *s;
        
        while i < s.len()
            invariant 
                0 <= i <= s.len(),
                s.len() == old_s.len(),
                forall|x: int| i <= x < s.len() ==> #[trigger] s[x] == old_s[x],
                forall|x: int| 0 <= x < i ==> #[trigger] s[x] == 2 * old_s[x],
                forall|x: int| 0 <= x < s.len() ==> #[trigger] old_s[x] >= -1073741824 && old_s[x] <= 1073741823,
            decreases s.len() - i,
        {
            let old_val = s[i];
            s.set(i, 2 * old_val);
            i = i + 1;
        }
    }
}

fn main() {}