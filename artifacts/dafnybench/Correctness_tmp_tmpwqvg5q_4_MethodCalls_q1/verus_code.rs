use vstd::prelude::*;

verus! {
    spec fn fusc(n: int) -> nat;

    proof fn rule1()
        ensures fusc(0) == 0nat
    {
        admit(); // axiom
    }

    proof fn rule2()
        ensures fusc(1) == 1nat
    {
        admit(); // axiom
    }

    proof fn rule3(n: nat)
        ensures fusc(2 * n as int) == fusc(n as int)
    {
        admit(); // axiom
    }

    proof fn rule4(n: nat)
        ensures fusc(2 * n as int + 1) == fusc(n as int) + fusc(n as int + 1)
    {
        admit(); // axiom
    }

    fn compute_fusc(N: int) -> (b: int)
        requires N >= 0
        ensures b == fusc(N)
    {
        let mut b: int = 0;
        let mut n: int = N;
        let mut a: int = 1;
        
        assert(0 <= n <= N);
        assert(fusc(N) == a * fusc(n) + b * fusc(n + 1));

        while n != 0
            invariant 
                0 <= n <= N,  // J1
                fusc(N) == a * fusc(n) + b * fusc(n + 1),  // J2
            decreases n  // D
        {
            let ghost d = n; // termination metric

            proof {
                assert(fusc(N) == a * fusc(n) + b * fusc(n + 1));
                assert(n != 0);

                // Weakest precondition reasoning
                assert((n % 2 != 0 && n % 2 == 0) || fusc(N) == a * fusc(n) + b * fusc(n + 1));
                assert((n % 2 != 0 || n % 2 == 0) ==> fusc(N) == a * fusc(n) + b * fusc(n + 1));

                assert(n % 2 != 0 || fusc(N) == a * fusc(n) + b * fusc(n + 1));
                assert(n % 2 == 0 || fusc(N) == a * fusc(n) + b * fusc(n + 1));
                
                assert(n % 2 == 0 ==> fusc(N) == a * fusc(n) + b * fusc(n + 1));
                assert(n % 2 != 0 ==> fusc(N) == a * fusc(n) + b * fusc(n + 1));
            }

            if n % 2 == 0 {
                // Even case
                proof {
                    rule4((n/2) as nat);
                    assert(fusc((n/2) + 1) == fusc(n + 1) - fusc(n/2));
                    
                    rule3((n/2) as nat);
                    assert(fusc(n/2) == fusc(n));
                    
                    assert(fusc(N) == (a + b) * fusc(n/2) + b * fusc((n/2) + 1));
                }
                
                a = a + b;
                
                proof {
                    assert(fusc(N) == a * fusc(n/2) + b * fusc((n/2) + 1));
                }
                
                n = n / 2;
                
                proof {
                    assert(fusc(N) == a * fusc(n) + b * fusc(n + 1));
                }
            } else {
                // Odd case  
                proof {
                    rule4(((n-1)/2) as nat);
                    assert(fusc(n) - fusc((n-1)/2) == fusc(((n-1)/2)+1));
                    
                    rule3(((n-1)/2) as nat);
                    assert(fusc((n-1)/2) == fusc(n-1));

                    assert(fusc(((n-1)/2)+1) == fusc((n+1)/2));
                    
                    rule3(((n+1)/2) as nat);
                    assert(fusc((n+1)/2) == fusc(n+1));

                    assert(fusc(N) == a * fusc(n) + b * fusc(n + 1));

                    assert(fusc(N) == b * fusc(((n-1)/2)+1) + a * fusc(n));

                    assert(fusc(N) ==
                            b * fusc(n) - b * fusc(n) + b * fusc(((n-1)/2)+1) + a * fusc(n));
                    
                    assert(fusc(N) ==
                            b * fusc(n) - b * (fusc(n) - fusc(((n-1)/2)+1)) + a * fusc(n));
                    
                    assert(fusc(N) == b * fusc(n) - b * fusc((n-1)/2) + a * fusc(n));
                    
                    assert(fusc(N) == b * fusc(n) - b * fusc(n-1) + a * fusc(n));
                    
                    assert(fusc(N) == b * fusc(n) - b * fusc(n-1) + a * fusc(n));
                    
                    assert(fusc(N) ==
                            a * fusc(n - 1) + b * fusc(n) - b * fusc(n-1) + a * fusc(n) - a * fusc(n-1));
                    assert(fusc(N) == a * fusc(n - 1) + (b + a) * (fusc(n) - fusc(n-1)));
             
                    assert(fusc(N) == a * fusc((n - 1)) + (b + a) * (fusc(n) - fusc((n-1)/2)));

                    assert(fusc(N) == a * fusc((n - 1) / 2) + (b + a) * fusc(((n - 1) / 2) + 1));
                }
                
                b = b + a;
                
                proof {
                    assert(fusc(N) == a * fusc((n - 1) / 2) + b * fusc(((n - 1) / 2) + 1));
                }
                
                n = (n - 1) / 2;

                proof {
                    assert(fusc(N) == a * fusc(n) + b * fusc(n + 1));
                }
            }
            
            proof {
                assert(n < d); // termination metric
                assert(fusc(N) == a * fusc(n) + b * fusc(n + 1)); // J
            }
        }
        
        // Post-loop assertions
        proof {
            assert(n == 0); // !B

            rule1();
            assert(fusc(0) == 0nat);

            rule2();
            assert(fusc(1) == 1nat);

            assert(fusc(N) == a * fusc(0) + b * fusc(1)); // J

            assert(fusc(N) == a * 0 + b * 1); // J
            assert(b == fusc(N));
        }
        
        b
    }

    fn main() {}
}