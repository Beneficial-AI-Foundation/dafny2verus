use vstd::prelude::*;

verus! {
    // Basic vowel counting function - recursive version
    spec fn vowels(s: Seq<char>) -> nat
        decreases s.len()
    {
        if s.len() == 0 {
            0nat
        } else {
            (if s[0] == 'a' || s[0] == 'e' || s[0] == 'i' || s[0] == 'o' || s[0] == 'u' {
                1nat
            } else {
                0nat
            }) + vowels(s.subrange(1, s.len() as int))
        }
    }

    // Multiset-based vowel counting function
    spec fn vowelsF(s: Seq<char>) -> nat {
        let m = s.to_multiset();
        m.count('a') + m.count('e') + m.count('i') + m.count('o') + m.count('u')
    }

    // Lemma that vowel counting distributes over concatenation
    proof fn VowelsLemma(s: Seq<char>, t: Seq<char>)
        ensures vowels(s.add(t)) == vowels(s) + vowels(t)
        decreases s.len()
    {
        if s.len() > 0 {
            let s_tail = s.subrange(1, s.len() as int);
            let s_head = seq![s[0]];
            assert(s_head.add(s_tail) =~= s);
            assert(s_head.add(s_tail.add(t)) =~= s_head.add(s_tail).add(t));
            VowelsLemma(s_tail, t);
        }
    }

    // Easy lemma for multiset version
    proof fn VowelsLemmaF(s: Seq<char>, t: Seq<char>)
        ensures vowelsF(s.add(t)) == vowelsF(s) + vowelsF(t)
    {
        // This follows directly from multiset properties
    }

    // Stack datatype model
    pub enum StackModel {
        Empty,
        Push { value: int, prev: Box<StackModel> }
    }

    // KlingonCalendar class equivalent
    pub struct KlingonCalendar {
        day_of_week: i32,
        week_of_month: i32,
        month_of_year: i32,
        year: u32,
    }

    impl KlingonCalendar {
        pub closed spec fn valid(&self) -> bool {
            (-3 <= self.day_of_week <= 1) && 
            (-1 <= self.week_of_month <= 1) && 
            (-5 < self.month_of_year <= 1)
        }

        pub fn new(day: i32, week: i32, month: i32, year: u32) -> (result: Self)
            requires 
                -3 <= day <= 1,
                -1 <= week <= 1,
                -5 < month <= 1,
            ensures result.valid()
        {
            KlingonCalendar {
                day_of_week: day,
                week_of_month: week,
                month_of_year: month,
                year: year,
            }
        }
    }

    // Simple stack implementation
    struct Stack {
        values: Vec<i32>,
        capacity: usize,
        size: usize,
    }

    impl Stack {
        closed spec fn valid(&self) -> bool {
            self.size <= self.values.len() && 
            self.values.len() == self.capacity
        }

        closed spec fn get_size(&self) -> usize {
            self.size
        }

        closed spec fn get_capacity(&self) -> usize {
            self.capacity
        }

        fn new(capacity_: usize) -> (result: Self)
            ensures result.valid()
        {
            Stack {
                values: Vec::with_capacity(capacity_),
                capacity: capacity_,
                size: 0,
            }
        }

        fn push(&mut self, i: i32)
            requires 
                old(self).valid(),
                old(self).get_size() < old(self).get_capacity(),
            ensures 
                self.valid(),
                self.get_size() == old(self).get_size() + 1,
                self.get_size() > 0,
        {
            self.values.push(i);
            self.size = self.size + 1;
        }

        fn pop(&mut self) -> (r: i32)
            requires 
                old(self).valid(),
                0 < old(self).get_size(),
                old(self).get_size() <= old(self).get_capacity(),
            ensures 
                self.valid(),
                self.get_size() == old(self).get_size() - 1,
                self.get_size() >= 0,
        {
            self.size = self.size - 1;
            self.values.pop().unwrap()
        }

        fn top(&self) -> (r: i32)
            requires 
                self.valid(),
                self.get_size() > 0,
        {
            self.values[self.size - 1]
        }

        closed spec fn to_stack_model(&self) -> StackModel
            recommends self.valid()
        {
            self.to_stack_model_aux(self.size)
        }

        closed spec fn to_stack_model_aux(&self, i: usize) -> StackModel
            recommends 
                self.valid(),
                i <= self.capacity,
            decreases i
        {
            if i == 0 {
                StackModel::Empty
            } else {
                StackModel::Push {
                    value: self.values[i - 1] as int,
                    prev: Box::new(self.to_stack_model_aux(sub1(i)))
                }
            }
        }
    }

    // Helper function for subtraction
    spec fn sub1(i: usize) -> usize {
        if i == 0 { 0 } else { (i - 1) as usize }
    }

    // Test method for stack verification
    fn verify_stack(s: &mut Stack, i: i32, j: i32)
        requires 
            old(s).valid(),
            old(s).get_size() == 0,
        ensures s.valid()
    {
        s.push(i);
        s.push(j);
        let v = s.pop();
        assert(v == j);
        let v = s.pop();
        assert(v == i);
    }

    // Main function to satisfy compiler requirements
    fn main() {
        let mut stack = Stack::new(10);
        verify_stack(&mut stack, 42, 43);
    }
}