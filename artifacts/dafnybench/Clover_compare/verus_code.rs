use vstd::prelude::*;

verus! {
    // For demonstration with a concrete type that supports equality
    fn compare_bool(a: bool, b: bool) -> (eq: bool)
        requires true,
        ensures
            a == b ==> eq == true,
            a != b ==> eq == false,
    {
        if a == b { true } else { false }
    }
    
    // For integers
    fn compare_int(a: int, b: int) -> (eq: bool)
        requires true,
        ensures
            a == b ==> eq == true,
            a != b ==> eq == false,
    {
        if a == b { true } else { false }
    }
    
    // For natural numbers
    fn compare_nat(a: nat, b: nat) -> (eq: bool)
        requires true,
        ensures
            a == b ==> eq == true,
            a != b ==> eq == false,
    {
        if a == b { true } else { false }
    }
}

fn main() {}