use vstd::prelude::*;

verus! {
    fn process_values(x: u32, y: u32) -> (r: (u32, u32, u32))
        requires
            x == y,
        ensures
            r.1 == r.2,
    {
        let mut x_out = x;
        let mut y_out = y;
        let n = 0u32;

        while x_out > n
            invariant
                x_out == y_out,
                x_out <= x,
            decreases x_out,
        {
            x_out = x_out - 1;
            y_out = y_out - 1;
        }

        (x_out, y_out, n)
    }
}

fn main() {}