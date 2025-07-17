use vstd::prelude::*;

verus! {
    struct TwoStacks<T> where T: std::default::Default {
        // Abstract state
        ghost s1: Seq<T>,
        ghost s2: Seq<T>,
        ghost n_const: nat, // maximum size of the stacks
        
        // Concrete state
        data: Vec<T>,
        n1: usize, // number of elements in stack 1
        n2: usize, // number of elements in stack 2
    }

    impl<T> TwoStacks<T> where T: std::default::Default {
        spec fn valid(&self) -> bool {
            &&& self.data.len() == self.n_const
            &&& self.n1 == self.s1.len()
            &&& self.n2 == self.s2.len()
            &&& self.n1 + self.n2 <= self.data.len()
        }

        fn new(n: usize) -> (result: Self)
            ensures 
                result.valid(),
                result.s1.len() == 0,
                result.s2.len() == 0,
                result.n_const == n,
        {
            let mut data = Vec::new();
            let mut i = 0;
            while i < n
                invariant 
                    i <= n,
                    data.len() == i,
            {
                data.push(T::default());
                i += 1;
            }
            
            TwoStacks {
                s1: Seq::empty(),
                s2: Seq::empty(), 
                n_const: proof { n as nat },
                data,
                n1: 0,
                n2: 0,
            }
        }

        fn push1(&mut self, element: T) -> (full_status: bool)
            requires old(self).valid(),
            ensures 
                self.valid(),
                full_status ==> self.s1.len() == old(self).s1.len() + 1,
                !full_status ==> self.s1 == old(self).s1,
        {
            if self.n1 + self.n2 >= self.data.len() {
                false
            } else {
                proof {
                    self.s1 = self.s1.push(element);
                }
                self.data.set(self.n1, element);
                self.n1 = self.n1 + 1;
                true
            }
        }

        fn push2(&mut self, element: T) -> (full_status: bool)
            requires old(self).valid(),
            ensures 
                self.valid(),
                full_status ==> self.s2.len() == old(self).s2.len() + 1,
                !full_status ==> self.s2 == old(self).s2,
        {
            if self.n1 + self.n2 >= self.data.len() {
                false
            } else {
                proof {
                    self.s2 = self.s2.push(element);
                }
                let idx = self.data.len() - 1 - self.n2;
                self.data.set(idx, element);
                self.n2 = self.n2 + 1;
                true
            }
        }

        fn pop1(&mut self) -> (result: (bool, T))
            requires old(self).valid()
            ensures 
                self.valid(),
                result.0 ==> old(self).s1.len() > 0,
                result.0 ==> self.s1.len() == old(self).s1.len() - 1,
                !result.0 ==> old(self).s1.len() == 0,
        {
            if self.n1 == 0 {
                (false, T::default())
            } else {
                proof {
                    self.s1 = self.s1.drop_last();
                }
                let popped_item = self.data[self.n1 - 1];
                self.n1 = self.n1 - 1;
                (true, popped_item)
            }
        }

        fn pop2(&mut self) -> (result: (bool, T))
            requires old(self).valid()
            ensures 
                self.valid(),
                result.0 ==> old(self).s2.len() > 0,
                result.0 ==> self.s2.len() == old(self).s2.len() - 1,
                !result.0 ==> old(self).s2.len() == 0,
        {
            if self.n2 == 0 {
                (false, T::default())
            } else {
                proof {
                    self.s2 = self.s2.drop_last();
                }
                let popped_item = self.data[self.data.len() - self.n2];
                self.n2 = self.n2 - 1;
                (true, popped_item)
            }
        }

        fn peek1(&self) -> (result: (bool, T))
            requires self.valid()
            ensures 
                result.0 ==> self.s1.len() > 0,
                !result.0 ==> self.s1.len() == 0,
        {
            if self.n1 == 0 {
                (false, T::default())
            } else {
                (true, self.data[self.n1 - 1])
            }
        }

        fn peek2(&self) -> (result: (bool, T))
            requires self.valid()
            ensures 
                result.0 ==> self.s2.len() > 0,
                !result.0 ==> self.s2.len() == 0,
        {
            if self.n2 == 0 {
                (false, T::default())
            } else {
                (true, self.data[self.data.len() - self.n2])
            }
        }

        spec fn empty1(&self) -> bool {
            self.s1.len() == 0
        }

        spec fn empty2(&self) -> bool {
            self.s2.len() == 0
        }
    }
}

fn main() {
    let mut stacks = TwoStacks::<i32>::new(10);
    let status1 = stacks.push1(5);
    let status2 = stacks.push2(10);
    let (empty_status, top) = stacks.peek1();
    println!("Stack operations completed: {}, {}, {}", status1, status2, empty_status);
}