use vstd::prelude::*;

verus! {
    // Predicate for primeness
    spec fn prime(n: nat) -> bool {
        n > 1 && (forall|nr: nat| 1 < nr < n ==> #[trigger] (n % nr) != 0)
    }

    // Datatype for Answer
    #[derive(PartialEq, Eq)]
    enum Answer {
        Yes,
        No,
        Unknown,
    }

    // Method to test whether a number is prime, returns bool
    fn test_primeness(n: u64) -> (result: bool)
        ensures result == prime(n as nat)
    {
        if n == 0 || n == 1 {
            return false;
        }
        let mut i: u64 = 2;
        let mut result = true;

        while i < n
            invariant 
                2 <= i <= n,
                result == (forall|j: nat| 1 < j < (i as nat) ==> #[trigger] ((n as nat) % j) != 0)
            decreases n - i
        {
            if n % i == 0 {
                result = false;
            }
            i = i + 1;
        }
        result
    }

    fn main() {
        let result = test_primeness(17);
        let result2 = test_primeness(15);
        let result3 = test_primeness(13);
    }
}