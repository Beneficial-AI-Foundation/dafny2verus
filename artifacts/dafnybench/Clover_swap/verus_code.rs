use vstd::prelude::*;

verus! {
    fn swap(X: int, Y: int) -> (result: (int, int))
        ensures 
            result.0 == Y,
            result.1 == X,
    {
        let (mut x, mut y) = (X, Y);
        
        let tmp = x;
        x = y;
        y = tmp;
        
        assert(x == Y && y == X);
        
        (x, y)
    }
}