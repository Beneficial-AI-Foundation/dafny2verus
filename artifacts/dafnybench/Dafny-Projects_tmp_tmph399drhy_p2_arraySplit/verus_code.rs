use vstd::prelude::*;

verus! {
    fn array_split(a: Vec<int>) -> (Vec<int>, Vec<int>)
    {
        let split_point: usize = a.len() / 2;
        
        let mut b = Vec::new();
        let mut c = Vec::new();
        
        // Fill the first half into b
        let mut i: usize = 0;
        while i < split_point
            invariant 
                0 <= i <= split_point <= a.len(),
                b.len() == i,
                b@ == a@.subrange(0, i as int),
            decreases split_point - i,
        {
            b.push(a[i]);
            i = i + 1;
        }
        
        // Fill the second half into c
        let mut j: usize = 0;
        while i < a.len()
            invariant 
                split_point <= i <= a.len(),
                j == i - split_point,
                c.len() == j,
                c@ == a@.subrange(split_point as int, i as int),
                split_point == b.len(),
                b@ == a@.subrange(0, split_point as int),
            decreases a.len() - i,
        {
            c.push(a[i]);
            i = i + 1;
            j = j + 1;
        }
        
        // Postcondition assertions - equivalent to ensures clauses
        assert(a.len() == b.len() + c.len());
        assert(a.len() > 1 ==> a.len() > b.len());
        assert(a.len() > 1 ==> a.len() > c.len());
        assert(a@ == b@ + c@);
        
        (b, c)
    }
}