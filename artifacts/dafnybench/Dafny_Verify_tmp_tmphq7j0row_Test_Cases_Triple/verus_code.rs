use vstd::prelude::*;

verus! {
    fn triple(x: u32) -> (r: u32)
        requires x <= 0x55555555u32,
        ensures r as int == 3 * (x as int)
    {
        let y = 2 * x;
        let r = x + y;
        assert(r as int == 3 * (x as int));
        r
    }

    fn triple_if(x: u32) -> (r: u32)
        requires x <= 0x55555555u32,
        ensures r as int == 3 * (x as int)
    {
        let r = if x == 0 {
            0
        } else {
            let y = 2 * x;
            x + y
        };
        assert(r as int == 3 * (x as int));
        r
    }

    fn triple_over(x: u32) -> (r: u32)
        requires x <= 0x55555555u32,
        ensures r as int == 3 * (x as int)
    {
        let r = if x < 18 {
            let a = 2 * x;
            let b = 4 * x;
            (a + b) / 2
        } else {
            let y = 2 * x;
            x + y
        };
        assert(r as int == 3 * (x as int));
        r
    }

    fn triple_conditions(x: u32) -> (r: u32)
        requires x % 2 == 0 && x <= 0x55555555u32,
        ensures r as int == 3 * (x as int)
    {
        let y = x / 2;
        let r = 6 * y;
        assert(r as int == 3 * (x as int));
        r
    }

    fn caller() {
        let t = triple_conditions(18);
        assert(t < 100);
    }
}

fn main() {}