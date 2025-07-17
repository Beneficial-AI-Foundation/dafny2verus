use vstd::prelude::*;

verus! {
    // Specification function for min
    spec fn min(a: int, b: int) -> int {
        if a < b { a } else { b }
    }

    // Proof function to establish properties of min
    proof fn min_properties(a: int, b: int)
        ensures 
            min(a, b) <= a && min(a, b) <= b,
            min(a, b) == a || min(a, b) == b,
    {
        // The properties follow from the definition
    }

    // Executable method for min with concrete types
    fn minMethod(a: i32, b: i32) -> (c: i32)
        ensures 
            c <= a && c <= b,
            c == a || c == b,
            c == min(a as int, b as int),
    {
        if a < b {
            a
        } else {
            b
        }
    }

    // Ghost function (spec function in Verus)
    spec fn minFunction(a: int, b: int) -> int {
        if a < b { a } else { b }
    }

    // Proof function to establish properties of minFunction
    proof fn minFunction_properties(a: int, b: int)
        ensures 
            minFunction(a, b) <= a && minFunction(a, b) <= b,
            minFunction(a, b) == a || minFunction(a, b) == b,
    {
        // The properties follow from the definition
    }

    // Method to find minimum in array
    fn minArray(a: &[i32]) -> (m: i32)
        requires a.len() > 0
        ensures 
            forall|k: int| 0 <= k < a.len() ==> m <= a[k],
            exists|k: int| 0 <= k < a.len() && m == a[k],
    {
        let mut m = a[0];
        let mut i = 1;
        
        while i < a.len()
            invariant 
                0 <= i <= a.len(),
                forall|k: int| 0 <= k < i ==> m <= a[k],
                exists|k: int| 0 <= k < i && m == a[k],
            decreases a.len() - i
        {
            if a[i] < m {
                m = a[i];
            }
            i = i + 1;
        }
        
        m
    }

    fn main() {
        let integer = minMethod(1, 2);
        // For demonstration purposes - shows the minimum value
        assert(integer == 1);
        
        // Test with array
        let arr = [3, 1, 4, 1, 5];
        let min_val = minArray(&arr);
        assert(min_val == 1);
    }
}