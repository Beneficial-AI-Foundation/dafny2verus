use vstd::prelude::*;

verus! {
    // Iterator-like struct to replace Dafny's iterator Gen
    pub struct Gen {
        pub start: u32,
        pub current: u32,
        pub count: usize,
        pub xs: Vec<u32>,
        pub finished: bool,
    }
    
    impl Gen {
        // Constructor equivalent to Dafny's iterator initialization
        pub fn new(start: u32) -> (result: Self)
            requires start <= 1000000,
            ensures 
                result.start == start,
                result.current == start,
                result.count == 0,
                result.xs@.len() == 0,
                result.finished == false,
        {
            Gen {
                start: start,
                current: start,
                count: 0,
                xs: Vec::new(),
                finished: false,
            }
        }
        
        // Spec function for validity - equivalent to Dafny's i.Valid()
        pub open spec fn valid_spec(&self) -> bool {
            self.count <= 10 && self.xs@.len() == self.count && self.start <= 1000000
        }
        
        // Check if iterator is valid
        pub fn valid(&self) -> (result: bool)
            ensures result == (self.count <= 10 && self.xs@.len() == self.count),
        {
            self.count <= 10 && self.xs.len() == self.count
        }
        
        // Move to next value - equivalent to Dafny's MoveNext()
        // Returns true if there's a next value, false if finished
        pub fn move_next(&mut self) -> (result: bool)
            requires old(self).valid_spec(),
            ensures 
                self.valid_spec(),
                result == (old(self).count < 10),
                if result {
                    // Yield ensures |xs| <= 10 && x == start + |xs| - 1
                    self.count == old(self).count + 1 &&
                    self.current == old(self).start + old(self).count as u32 &&
                    self.xs@.len() == old(self).xs@.len() + 1 &&
                    self.xs@[self.xs@.len() - 1] == self.current &&
                    self.xs@.len() <= 10 &&
                    self.current == self.start + (self.xs@.len() - 1) as u32
                } else {
                    self.finished == true &&
                    self.count == old(self).count &&
                    self.xs@ == old(self).xs@
                },
        {
            if self.count < 10 {
                self.current = self.start + self.count as u32;
                self.xs.push(self.current);
                self.count = self.count + 1;
                true
            } else {
                self.finished = true;
                false
            }
        }
        
        // Get current value (equivalent to i.x in Dafny)
        pub fn get_current(&self) -> u32 {
            self.current
        }
        
        // Get the sequence of yielded values so far (equivalent to i.xs in Dafny)
        pub fn get_xs(&self) -> (result: &Vec<u32>)
            ensures result@ == self.xs@,
        {
            &self.xs
        }
    }
    
    // Main function equivalent
    pub fn main() {
        let mut i = Gen::new(30);
        
        // Bounded loop equivalent to the original infinite loop
        let mut iterations = 0usize;
        while iterations < 10 && i.valid()
            invariant 
                i.valid_spec(),
                iterations <= 10,
                i.count <= 10,
            decreases 10 - i.count,
        {
            let m = i.move_next();
            if !m {
                break;
            }
            // In actual implementation, you would print i.get_current()
            // This corresponds to "print i.x;" in the original Dafny
            iterations = iterations + 1;
        }
    }
}