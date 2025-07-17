use vstd::prelude::*;

verus! {
    fn insert(line: &mut Vec<char>, l: usize, nl: &Vec<char>, p: usize, at: usize)
        requires 
            l + p <= old(line).len(),
            p <= nl.len(),
            at <= l,
            l <= old(line).len(),
        ensures 
            line.len() == old(line).len(),
            forall|i: int| (0 <= i < p) ==> line[at + i] == nl[i],
            forall|i: int| (0 <= i < at) ==> line[i] == old(line)[i],
            forall|i: int| (at + p <= i < l + p) ==> line[i] == old(line)[i - p],
    {
        let ghost initial_line = *line;
        
        // First loop: shift elements to the right
        let mut i: usize = l;
        while i > at
            invariant
                at <= i <= l,
                i + p <= line.len(),
                line.len() == initial_line.len(),
                // Parts of the array that haven't been modified yet
                forall|j: int| (0 <= j < i) ==> line[j] == initial_line[j],
                // Parts that have been shifted 
                forall|j: int| (i + p <= j < l + p && j < line.len()) ==> line[j] == initial_line[j - p],
            decreases i
        {
            i = i - 1;
            let tmp = line[i];
            line.set(i + p, tmp);
        }
        
        // Second loop: insert the new elements
        i = 0;
        while i < p
            invariant
                0 <= i <= p,
                at + p <= line.len(),
                i <= nl.len(),
                line.len() == initial_line.len(),
                line.len() == old(line).len(),
                p <= nl.len(),
                l + p <= old(line).len(),
                // Parts before the insertion point are unchanged
                forall|j: int| (0 <= j < at) ==> line[j] == initial_line[j],
                // Parts that have been inserted so far
                forall|j: int| (at <= j < at + i) ==> line[j] == nl[j - at],
                // Parts that have been shifted (only if they're valid indices)
                forall|j: int| (at + p <= j < l + p && j < line.len() && j - p >= 0 && j - p < initial_line.len()) ==> line[j] == initial_line[j - p],
            decreases p - i
        {
            line.set(at + i, nl[i]);
            i = i + 1;
        }
    }
}