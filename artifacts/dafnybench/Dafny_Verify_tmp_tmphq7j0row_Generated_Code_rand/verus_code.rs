use vstd::prelude::*;

verus! {
    fn main_method(x_init: u32, y: u32) -> (z: u32)
        requires 
            (x_init as int) * (y as int) <= u32::MAX as int,
        ensures z == 0
    {
        let mut x = x_init;
        let mut z = x * y;
        
        while x > 0
            invariant 
                (x as int) * (y as int) == z as int,
                x <= x_init,
                (x as int) * (y as int) <= u32::MAX as int,
                y == 0 || z >= y,
            decreases x
        {
            proof {
                if y > 0 {
                    // Since x >= 1 and y > 0, we have z = x * y >= y
                    assert(x as int >= 1);
                    assert((x as int) * (y as int) >= 1 * (y as int));
                    assert(1 * (y as int) == y as int);
                    assert(z as int >= y as int);
                    assert(z >= y);
                }
                
                // Prove the invariant is maintained after the loop body
                vstd::arithmetic::mul::lemma_mul_is_distributive_sub(x as int, 1int, y as int);
                assert((x as int - 1) * (y as int) == (x as int) * (y as int) - (y as int));
            }
            
            x = x - 1;
            z = z - y;
        }
        
        z
    }
}

fn main() {}