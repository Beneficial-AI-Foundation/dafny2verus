use vstd::prelude::*;

verus! {
    // Predicate to check if a number is odd
    spec fn is_odd(x: int) -> bool {
        x % 2 == 1
    }

    // Newtype for odd numbers
    #[derive(PartialEq, Eq, Clone, Copy, Debug)]
    struct Odd {
        value: i32,
    }

    impl Odd {
        spec fn as_int(self) -> int {
            self.value as int
        }
    }

    // OddList implementation
    struct OddList {
        vec: Vec<Odd>,
        cap: usize,
    }

    impl OddList {
        spec fn view(self) -> Seq<Odd> {
            self.vec.view()
        }

        spec fn capacity(self) -> nat {
            self.cap as nat
        }

        spec fn valid(self) -> bool {
            0 <= self.vec.len() <= self.capacity() &&
            forall|i: int| 0 <= i < self.vec.len() ==> is_odd(self.vec[i].as_int())
        }

        fn new(capacity: usize) -> (result: Self)
            ensures
                result.valid(),
                result.view().len() == 0,
                result.capacity() == capacity as nat,
        {
            OddList {
                vec: Vec::new(),
                cap: capacity,
            }
        }

        fn insert(&mut self, index: usize, element: Odd)
            requires
                index <= old(self).view().len(),
                old(self).view().len() + 1 <= old(self).capacity(),
                old(self).valid(),
                is_odd(element.as_int()),
            ensures
                self.view().len() == old(self).view().len() + 1,
                self.view()[index as int] == element,
                self.capacity() == old(self).capacity(),
                self.valid(),
        {
            self.vec.insert(index, element);
        }

        fn push_front(&mut self, element: Odd)
            requires 
                old(self).view().len() + 1 <= old(self).capacity(),
                old(self).valid(),
                is_odd(element.as_int()),
            ensures
                self.view().len() == old(self).view().len() + 1,
                self.view()[0] == element,
                self.capacity() == old(self).capacity(),
                self.valid(),
        {
            self.insert(0, element);
        }

        fn push_back(&mut self, element: Odd)
            requires 
                old(self).view().len() + 1 <= old(self).capacity(),
                old(self).valid(),
                is_odd(element.as_int()),
            ensures
                self.view().len() == old(self).view().len() + 1,
                self.view()[self.view().len() - 1] == element,
                self.capacity() == old(self).capacity(),
                self.valid(),
        {
            let len = self.vec.len();
            self.insert(len, element);
        }

        fn remove_at_index(&mut self, index: usize)
            requires
                old(self).valid(),
                old(self).view().len() > 0,
                index < old(self).view().len(),
            ensures
                self.view().len() == old(self).view().len() - 1,
                self.capacity() == old(self).capacity(),
                self.valid(),
        {
            self.vec.remove(index);
        }

        fn pop_front(&mut self) -> (x: Odd)
            requires
                old(self).valid(),
                old(self).view().len() > 0,
            ensures
                old(self).view()[0] == x,
                self.view().len() == old(self).view().len() - 1,
                self.capacity() == old(self).capacity(),
                self.valid(),
        {
            let x = self.vec[0];
            self.vec.remove(0);
            x
        }

        fn pop_back(&mut self) -> (x: Odd)
            requires
                old(self).valid(),
                old(self).view().len() > 0,
            ensures
                old(self).view()[old(self).view().len() - 1] == x,
                self.view().len() == old(self).view().len() - 1,
                self.capacity() == old(self).capacity(),
                self.valid(),
        {
            let last_index = self.vec.len() - 1;
            let x = self.vec[last_index];
            self.vec.pop();
            x
        }

        fn length(&self) -> (n: usize)
            ensures n == self.view().len(),
        {
            self.vec.len()
        }

        fn at(&self, index: usize) -> (x: Odd)
            requires index < self.view().len(),
            ensures x == self.view()[index as int],
        {
            self.vec[index]
        }
    }
}

fn main() {
    let mut list = OddList::new(10);
    let odd_3 = Odd { value: 3 };
    let odd_5 = Odd { value: 5 };
    
    list.push_back(odd_3);
    list.push_back(odd_5);
    
    println!("Length: {}", list.length());
    println!("First element: {:?}", list.at(0));
    println!("Second element: {:?}", list.at(1));
}