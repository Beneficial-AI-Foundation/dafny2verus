use vstd::prelude::*;

verus! {
    fn SwapSimultaneous(X: int, Y: int) -> (result: (int, int))
        ensures
            result.0 == Y,
            result.1 == X,
    {
        let (x, y) = (X, Y);
        let (x, y) = (y, x);
        (x, y)
    }
}

fn main() {}