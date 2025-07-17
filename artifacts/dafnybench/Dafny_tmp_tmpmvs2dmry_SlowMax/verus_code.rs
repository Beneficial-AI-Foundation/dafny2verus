use vstd::prelude::*;

verus! {
    spec fn max(x: nat, y: nat) -> nat {
        if x < y { y } else { x }
    }

    fn slow_max(a: u32, b: u32) -> (z: u32)
        requires a < 0x80000000 && b < 0x80000000
        ensures z == max(a as nat, b as nat)
    {
        let mut z: u32 = 0;
        let mut x: u32 = a;
        let mut y: u32 = b;
        
        while z < x && z < y
            invariant
                x >= 0,
                y >= 0,
                z == a - x && z == b - y,
                a - x == b - y,
            decreases x, y
        {
            z = z + 1;
            x = x - 1;
            y = y - 1;
        }

        if x <= y { 
            b 
        } else { 
            a 
        }
    }
}