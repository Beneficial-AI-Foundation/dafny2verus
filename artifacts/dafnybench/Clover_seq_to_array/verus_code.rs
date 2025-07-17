use vstd::prelude::*;

verus! {
    fn to_array<T>(xs: Seq<T>) -> (a: Vec<T>)
        ensures
            a.len() == xs.len(),
            a.view() == xs,
    {
        // In Verus, we use Vec<T> instead of array<T> since arrays have fixed compile-time size
        // The closest equivalent to the Dafny array creation with initialization function
        // is to build the vector element by element
        let mut result = Vec::new();
        let mut i = 0;
        while i < xs.len()
            invariant
                0 <= i <= xs.len(),
                result.len() == i,
                result.view() == xs.take(i as int),
        {
            result.push(xs[i as int]);
            i += 1;
        }
        result
    }
}