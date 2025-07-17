use vstd::prelude::*;

verus! {
    fn copy_impl(src: Vec<i32>, s_start: usize, dest: Vec<i32>, d_start: usize, len: usize) -> (r: Vec<i32>)
        requires 
            src.len() >= s_start + len,
            dest.len() >= d_start + len,
        ensures
            r.len() == dest.len(),
            r@.subrange(0, d_start as int) == dest@.subrange(0, d_start as int),
            d_start + len <= dest.len() ==> r@.subrange((d_start + len) as int, dest.len() as int) == dest@.subrange((d_start + len) as int, dest.len() as int),
            r@.subrange(d_start as int, (d_start + len) as int) == src@.subrange(s_start as int, (s_start + len) as int),
    {
        if len == 0 { 
            return dest; 
        }
        
        let mut i: usize = 0;
        let mut r = dest.clone();
        
        i = 0;
        while i < len
            invariant 
                i <= len,
                r.len() == dest.len(),
                d_start + len <= dest.len(),
                s_start + len <= src.len(),
                r@.subrange(0, d_start as int) == dest@.subrange(0, d_start as int),
                d_start + len <= dest.len() ==> r@.subrange((d_start + len) as int, dest.len() as int) == dest@.subrange((d_start + len) as int, dest.len() as int),
                forall |j: int| d_start <= j < d_start + i ==> r@[j] == src@[s_start + (j - d_start)],
            decreases len - i,
        {
            assert(i < len);
            assert(d_start + i < d_start + len);
            assert(s_start + i < s_start + len);
            
            r.set(d_start + i, src[s_start + i]);
            
            assert(forall |j: int| d_start <= j < d_start + i ==> r@[j] == src@[s_start + (j - d_start)]);
            assert(r@[d_start + i] == src@[s_start + i]);
            assert(forall |j: int| d_start <= j < d_start + (i + 1) ==> r@[j] == src@[s_start + (j - d_start)]);
            
            i = i + 1;
        }
        
        r
    }
}

fn main() {}