use vstd::prelude::*;

verus! {
    // Translation of the original Dafny method
    // Note: Original used nondeterministic choice (*), this version uses a parameter for choice
    fn main_method(n: u32, choices: Seq<bool>) -> (result: (u32, u32))
        requires 
            n >= 0,
            n <= 1000,  // Conservative bound to prevent overflow
            choices.len() == n as int,
        ensures result.0 + result.1 == 3 * n,
    {
        let mut i = 0u32;
        let mut a = 0u32;
        let mut b = 0u32;

        while i < n
            invariant 
                0 <= i <= n,
                a + b == 3 * i,
                a <= 2 * n,
                b <= 2 * n,
            decreases n - i,
        {
            // Original Dafny code had nondeterministic choice:
            // if(*) { a := a + 1; b := b + 2; }
            // else { a := a + 2; b := b + 1; }
            // Both branches maintain the invariant by adding 3 to (a + b)
            
            // We parametrize the choice through the choices sequence
            let ghost choice = choices[i as int];
            if choice {
                a = a + 1;
                b = b + 2;
            } else {
                a = a + 2;
                b = b + 1;
            }

            i = i + 1;
        }

        (a, b)
    }
}