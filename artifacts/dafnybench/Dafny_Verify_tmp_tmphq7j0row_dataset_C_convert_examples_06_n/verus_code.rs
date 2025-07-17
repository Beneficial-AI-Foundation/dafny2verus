use vstd::prelude::*;

verus! {
    #[verifier::exec_allows_no_decreases_clause]
    fn main_method() -> (ret: (u32, u32))
        ensures ret.0 == ret.1
    {
        let mut x: u32 = 0;
        let mut y: u32 = 0;
        let mut w: u32 = 1;
        let mut z: u32 = 0;
        let mut turn: u32 = 0;

        while x != y
            invariant 
                x < 1000,
                y < 1000,
                w < 1000,
                z < 1000,
                turn < 3
        {
            if turn == 0 {
                turn = 1;
            }

            if turn == 1 {
                if w % 2 == 1 && x < 999 {
                    x = x + 1;
                }

                if z % 2 == 0 && y < 999 {
                    y = y + 1;
                }

                turn = 1;
            } else {
                if turn == 2 {
                    if z + y < 999 {
                        z = z + y;
                        w = z + 1;
                    }

                    turn = 0;
                }
            }
        }

        (x, y)
    }
}

fn main() {
    println!("Verus translation compiled successfully!");
}