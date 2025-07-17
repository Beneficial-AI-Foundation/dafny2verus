use vstd::prelude::*;

verus! {
    spec fn hash(s: Seq<char>) -> int {
        sum_chars(s) % 137
    }

    spec fn sum_chars(s: Seq<char>) -> int 
        decreases s.len()
    {
        if s.len() == 0 {
            0
        } else {
            s[s.len() - 1] as int + sum_chars(s.subrange(0, s.len() - 1))
        }
    }

    struct CheckSumCalculator {
        cs: u32,
    }

    impl CheckSumCalculator {
        fn new() -> (result: CheckSumCalculator)
            ensures result.cs == 0
        {
            CheckSumCalculator { cs: 0 }
        }

        fn append_single(&mut self, c: char)
            ensures self.cs < 137
        {
            let char_val = c as u32;
            let sum = self.cs.wrapping_add(char_val);
            self.cs = sum % 137;
        }

        fn checksum(&self) -> u32 {
            self.cs
        }
    }

    fn main() {
        let mut calc = CheckSumCalculator::new();
        calc.append_single('a');
        calc.append_single('b');
        let result = calc.checksum();
    }
}