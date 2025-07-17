use vstd::prelude::*;

verus! {
    // Helper function to create a slice of a Vec<char>
    fn slice_vec(v: &Vec<char>, start: usize, end: usize) -> (result: Vec<char>)
        requires start <= end && end <= v.len(),
        ensures result.len() == end - start,
    {
        let mut result = Vec::new();
        let mut i = start;
        while i < end
            invariant 
                start <= i <= end,
                result.len() == i - start,
                end <= v.len(),
            decreases end - i,
        {
            result.push(v[i]);
            i = i + 1;
        }
        result
    }

    // This method should return true iff pre is a prefix of str. That is, str starts with pre
    fn is_prefix(pre: &Vec<char>, str: &Vec<char>) -> (res: bool)
        requires 0 < pre.len() <= str.len(), // This line states that this method requires that pre is less than or equal in length to str
    {
        // Initialising the index variable
        let mut i: usize = 0;

        // Iterating through the first |pre| elements in str
        while i < pre.len()
            invariant 
                0 <= i <= pre.len(), // Specifying the range of the while loop
                pre.len() <= str.len(), // Needed for bounds checking
            decreases pre.len() - i, // Specifying that the while loop will terminate
        {
            // If an element does not match, return false
            if str[i] != pre[i] {
                // Return once mismatch detected, no point in iterating any further
                return false;
            }
            // Else loop until mismatch found or we have reached the end of pre
            else {
                i = i + 1;
            }
        }
        true
    }

    // This method should return true iff sub is a substring of str. That is, str contains sub
    fn is_substring(sub: &Vec<char>, str: &Vec<char>) -> (res: bool)
        requires 0 < sub.len() <= str.len(), // This method requires that sub is less than or equal in length to str
    {
        // Initialising the index variable
        let mut i: usize = 0;

        // This variable stores the difference in length between the two strings
        let n: usize = str.len() - sub.len();

        // Here, we want to re-use the "is_prefix" method above, so with each iteration of the search, 
        // we are passing an offset of str - effectively trimming a character off the front of str and passing it to is_prefix
        while i <= n
            invariant 
                0 <= i <= n + 1, // Specifying the range of the while loop
                n + sub.len() == str.len(),
                sub.len() <= str.len(),
                0 < sub.len(),
            decreases n - i, // Specifying that the while loop will terminate
        {
            let str_slice = slice_vec(str, i, str.len());
            assert(str_slice.len() == str.len() - i);
            assert(str_slice.len() >= sub.len());
            let result = is_prefix(sub, &str_slice);

            // Return once the substring is found, no point in iterating any further
            if result == true {
                return true;
            }
            // Else loop until sub is found, or we have reached the end of str
            else {
                if i < n {
                    i = i + 1;
                } else {
                    break;
                }
            }
        }
        false
    }

    // This method should return true iff str1 and str2 have a common substring of length k
    fn have_common_k_substring(k: usize, str1: &Vec<char>, str2: &Vec<char>) -> (found: bool)
        requires 
            0 < k <= str1.len() && 0 < k <= str2.len(), // This method requires that k > 0 and k is less than or equal to in length to str1 and str2
    {
        // Initialising the index variable
        let mut i: usize = 0;

        // This variable is used to define the end condition of the while loop
        let n: usize = str1.len() - k;

        // Here, we want to re-use the "is_substring" method above, so with each iteration of the search, 
        // we are passing a substring of str1 with length k and searching for this substring in str2
        while i <= n
            invariant 
                0 <= i <= n + 1, // Specifying the range of the while loop
                n + k == str1.len(),
                k <= str2.len(),
                0 < k,
            decreases n - i, // Specifying that the loop will terminate
        {
            let str1_slice = slice_vec(str1, i, i + k);
            assert(str1_slice.len() == k);
            let result = is_substring(&str1_slice, str2);

            // Return once the length-k substring is found, no point in iterating any further
            if result == true {
                return true;
            }
            // Else loop until the length-k substring is found, or we have reached the end condition
            else {
                if i < n {
                    i = i + 1;
                } else {
                    break;
                }
            }
        }
        false
    }

    // This method should return the natural number len which is equal to the length of the longest common substring of str1 and str2
    fn max_common_substring_length(str1: &Vec<char>, str2: &Vec<char>) -> (len: usize)
        requires 0 < str1.len() && 0 < str2.len(),
    {
        // This variable is used to store the result of calling have_common_k_substring
        let mut result: bool;
        
        // We want the longest common substring between str1 and str2, so the starting point is going to be the shorter of the two strings
        let mut i: usize = str1.len();
        if str2.len() < str1.len() {
            i = str2.len();
        }

        // Here, we want to re-use the "have_common_k_substring" method above, so with each iteration of the search, 
        // we pass a decreasing value of k until a common substring of this length is found
        while i > 0
            invariant 0 <= i <= str1.len() && 0 <= i <= str2.len(),
            decreases i,
        {
            result = have_common_k_substring(i, str1, str2);

            if result == true {
                return i;
            }
            else {
                i = i - 1;
            }
        }
        0
    }

    // Main to test each method
    fn main() {
        // Test strings - convert to Vec<char>
        let string1 = vec!['o', 'p', 'e', 'r', 'a', 't', 'i', 'o', 'n'];
        let string2 = vec!['i', 'r', 'r', 'a', 't', 'i', 'o', 'n', 'a', 'l'];
        
        let x = max_common_substring_length(&string1, &string2);
        // Result would be printed here in a real implementation
    }
}