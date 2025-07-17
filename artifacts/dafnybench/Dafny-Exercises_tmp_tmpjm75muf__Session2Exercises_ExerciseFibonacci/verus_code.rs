use vstd::prelude::*;

verus! {
    spec fn fib(n: nat) -> nat
        decreases n
    {
        if n == 0 {
            0
        } else if n == 1 {
            1
        } else {
            fib((n - 1) as nat) + fib((n - 2) as nat)
        }
    }

    fn fibonacci1(n: u32) -> (f: u32)
        requires n <= 10,  
        ensures f == fib(n as nat)
    {
        let mut i: u32 = 0;
        let mut f: u32 = 0;
        let mut fsig: u32 = 1;
        
        while i < n
            invariant 
                f == fib(i as nat),
                fsig == fib((i + 1) as nat),
                i <= n
            decreases n - i
        {
            let temp = f + fsig;
            f = fsig;
            fsig = temp;
            i = i + 1;
        }
        f
    }

    fn fibonacci2(n: u32) -> (f: u32) 
        requires n <= 10,
        ensures f == fib(n as nat)
    {
        if n == 0 {
            0
        } else {
            let mut i: u32 = 1;
            let mut fant: u32 = 0;
            let mut f: u32 = 1;
            
            while i < n
                invariant 
                    fant == fib((i - 1) as nat),
                    f == fib(i as nat),
                    i <= n,
                    i >= 1
                decreases n - i
            {
                let temp = fant + f;
                fant = f;
                f = temp;
                i = i + 1;
            }
            f
        }
    }

    fn fibonacci3(n: u32) -> (f: u32)
        requires n <= 10,
        ensures f == fib(n as nat)
    {
        let mut i: u32 = 0;
        let mut a: u32 = 1;
        let mut f: u32 = 0;
        
        while i < n
            invariant 
                0 <= i <= n,
                if i == 0 {
                    a == fib(1) && f == fib(0)
                } else {
                    a == fib((i - 1) as nat) && f == fib(i as nat)
                }
            decreases n - i
        {
            let temp = a + f;
            a = f;
            f = temp;
            i = i + 1;
        }
        f
    }
}