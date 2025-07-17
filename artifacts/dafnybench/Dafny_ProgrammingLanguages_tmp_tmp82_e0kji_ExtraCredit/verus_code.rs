use vstd::prelude::*;

verus! {
    #[derive(PartialEq, Eq)]
    pub enum Exp {
        Const(int),
        Var(String),
        Plus(Box<Exp>, Box<Exp>),
        Mult(Box<Exp>, Box<Exp>),
    }

    spec fn eval(e: Exp, store: Map<String, int>) -> int
        decreases e
    {
        match e {
            Exp::Const(n) => n,
            Exp::Var(s) => if store.contains_key(s) { store[s] } else { -1 },
            Exp::Plus(e1, e2) => eval(*e1, store) + eval(*e2, store),
            Exp::Mult(e1, e2) => eval(*e1, store) * eval(*e2, store),
        }
    }

    spec fn optimize(e: Exp) -> Exp
        decreases e
    {
        match e {
            Exp::Mult(e1, e2) => match (*e1, *e2) {
                (Exp::Const(n), _) if n == 0 => Exp::Const(0),
                (_, Exp::Const(n)) if n == 0 => Exp::Const(0),
                (Exp::Const(n), e) if n == 1 => e,
                (e, Exp::Const(n)) if n == 1 => e,
                (Exp::Const(n1), Exp::Const(n2)) => Exp::Const(n1 * n2),
                _ => e,
            },
            Exp::Plus(e1, e2) => match (*e1, *e2) {
                (Exp::Const(n), e) if n == 0 => e,
                (e, Exp::Const(n)) if n == 0 => e,
                (Exp::Const(n1), Exp::Const(n2)) => Exp::Const(n1 + n2),
                _ => e,
            },
            _ => e,
        }
    }

    proof fn optimize_correct(e: Exp, s: Map<String, int>)
        ensures eval(e, s) == eval(optimize(e), s)
        decreases e
    {
        // Proof implementation with explicit arithmetic reasoning
        match e {
            Exp::Mult(e1, e2) => {
                let v1 = eval(*e1, s);
                let v2 = eval(*e2, s);
                assert(eval(e, s) == v1 * v2);
                
                match (*e1, *e2) {
                    (Exp::Const(n), _) if n == 0 => {
                        assert(v1 == 0);
                        assert(eval(e, s) == 0 * v2);
                        assert(0 * v2 == 0);
                        assert(eval(optimize(e), s) == 0);
                    },
                    // ... more cases
                }
            },
            // ... other cases
        }
    }

    proof fn optimize_simple_tests() {
        // Test assertions for optimization correctness
        assert(optimize(Exp::Mult(Box::new(Exp::Const(3)), Box::new(Exp::Const(4)))) == Exp::Const(12));
        // ... more tests
    }
}