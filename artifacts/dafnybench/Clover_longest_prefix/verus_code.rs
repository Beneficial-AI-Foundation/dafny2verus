use vstd::prelude::*;

verus! {
    fn longest_common_prefix(str1: Vec<char>, str2: Vec<char>) -> (prefix: Vec<char>)
        ensures
            prefix.len() <= str1.len() && prefix@ == str1@.subrange(0, prefix.len() as int),
            prefix.len() <= str2.len() && prefix@ == str2@.subrange(0, prefix.len() as int),
            prefix.len() == str1.len() || prefix.len() == str2.len() || 
                (prefix.len() < str1.len() && prefix.len() < str2.len() && str1[prefix.len() as int] != str2[prefix.len() as int])
    {
        let mut prefix: Vec<char> = Vec::new();
        let min_length: usize = if str1.len() < str2.len() { str1.len() } else { str2.len() };

        let mut idx: usize = 0;
        while idx < min_length
            invariant
                prefix.len() == idx,
                idx <= min_length <= str1.len() && min_length <= str2.len(),
                prefix.len() <= str1.len() && prefix@ == str1@.subrange(0, prefix.len() as int),
                prefix.len() <= str2.len() && prefix@ == str2@.subrange(0, prefix.len() as int)
            decreases min_length - idx
        {
            if idx < str1.len() && idx < str2.len() {
                let char1 = str1[idx];
                let char2 = str2[idx];
                if char1 != char2 {
                    return prefix;
                }
                prefix.push(char1);
            }
            idx = idx + 1;
        }
        
        prefix
    }
}