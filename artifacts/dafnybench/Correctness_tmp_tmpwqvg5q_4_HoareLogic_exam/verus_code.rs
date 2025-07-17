use vstd::prelude::*;

verus! {
    // Uninterpreted function for gcd - we assume its properties through axioms
    spec fn gcd(a: nat, b: nat) -> nat;

    // Lemma r1: gcd(a, 0) == a
    proof fn r1(a: nat)
        ensures gcd(a, 0) == a
    {
        // This is an axiom/assumption about gcd
        assume(gcd(a, 0) == a);
    }

    // Lemma r2: gcd(a, a) == a
    proof fn r2(a: nat)
        ensures gcd(a, a) == a
    {
        // This is an axiom/assumption about gcd
        assume(gcd(a, a) == a);
    }

    // Lemma r3: gcd(a, b) == gcd(b, a) (commutativity)
    proof fn r3(a: nat, b: nat)
        ensures gcd(a, b) == gcd(b, a)
    {
        // This is an axiom/assumption about gcd
        assume(gcd(a, b) == gcd(b, a));
    }

    // Lemma r4: b > 0 ==> gcd(a, b) == gcd(b, a % b) (Euclidean property)
    proof fn r4(a: nat, b: nat)
        ensures b > 0 ==> gcd(a, b) == gcd(b, a % b)
    {
        // This is an axiom/assumption about gcd
        assume(b > 0 ==> gcd(a, b) == gcd(b, a % b));
    }

    fn GCD1(a: u32, b: u32) -> (r: u32)
        requires a > 0 && b > 0,
        ensures gcd(a as nat, b as nat) == r,
        decreases b
    {
        if a < b {
            proof { r3(a as nat, b as nat); }
            GCD1(b, a)
        } else if (a % b) == 0 {
            proof {
                r4(a as nat, b as nat);
                assert(b > 0);
                assert(gcd(a as nat, b as nat) == gcd(b as nat, (a % b) as nat));
                assert(a % b == 0);
                assert(gcd(a as nat, b as nat) == gcd(b as nat, 0));
                r1(b as nat);
                assert(gcd(a as nat, b as nat) == b);
            }
            let r = b;
            proof { assert(gcd(a as nat, b as nat) == r); }
            r
        } else {
            proof { r4(a as nat, b as nat); }
            let r = GCD1(b, a % b);
            proof { assert(gcd(a as nat, b as nat) == r); }
            r
        }
    }

    fn GCD2(a: u32, b: u32) -> (r: u32)
        requires a > 0 && b >= 0,
        ensures gcd(a as nat, b as nat) == r,
        decreases b
    {
        proof {
            r1(a as nat);
            r4(a as nat, b as nat);
            assert(
                (b != 0 || (a > 0 && b >= 0 && gcd(a as nat, b as nat) == a))
                &&
                ((b == 0) || (b > 0 && (a % b) >= 0 ==> gcd(a as nat, b as nat) == gcd(b as nat, (a % b) as nat)))
            );
            assert(b != 0 || (a > 0 && b >= 0 && gcd(a as nat, b as nat) == a));
            assert(b == 0 ==> a > 0 && b >= 0 && gcd(a as nat, b as nat) == a);
            assert((b == 0) || (b > 0 && (a % b) >= 0 ==> gcd(a as nat, b as nat) == gcd(b as nat, (a % b) as nat)));
            assert(b >= 0 && b != 0 ==> b > 0 && (a % b) >= 0 ==> gcd(a as nat, b as nat) == gcd(b as nat, (a % b) as nat));
        }
        
        if b == 0 {
            proof {
                r1(a as nat);
                assert(gcd(a as nat, b as nat) == a);
            }
            let r = a;
            proof { assert(gcd(a as nat, b as nat) == r); }
            r
        } else {
            proof {
                r4(a as nat, b as nat);
                // Method call rule
                assert(b > 0 && (a % b) >= 0 ==> gcd(a as nat, b as nat) == gcd(b as nat, (a % b) as nat));
            }
            let r = GCD2(b, a % b);
            proof { assert(gcd(a as nat, b as nat) == r); }
            r
        }
    }
}

fn main() {}