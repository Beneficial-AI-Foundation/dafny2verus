use vstd::prelude::*;

verus! {
    fn SwapArithmetic(X: int, Y: int) -> (ret: (int, int))
        ensures
            ret.0 == Y,
            ret.1 == X,
    {
        let mut x = X;
        let mut y = Y;

        x = y - x;
        y = y - x;
        x = y + x;

        (x, y)
    }
}

fn main() {}