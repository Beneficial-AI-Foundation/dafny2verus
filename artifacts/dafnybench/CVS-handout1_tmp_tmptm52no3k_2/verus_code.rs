use vstd::prelude::*;

verus! {
    // Functional Lists datatype
    pub enum List<T> {
        Nil,
        Cons { head: T, tail: Box<List<T>> }
    }

    // Length function
    pub open spec fn length<T>(l: &List<T>) -> nat
        decreases l
    {
        match l {
            List::Nil => 0,
            List::Cons { head: _, tail } => 1 + length(tail)
        }
    }

    // Membership predicate
    pub open spec fn mem<T>(l: &List<T>, x: T) -> bool
        decreases l
    {
        match l {
            List::Nil => false,
            List::Cons { head, tail } => if *head == x { true } else { mem(tail, x) }
        }
    }

    // Access element at index
    pub open spec fn at<T>(l: &List<T>, i: nat) -> T
        decreases l
    {
        match l {
            List::Cons { head, tail } => {
                if i == 0 { *head } else { at(tail, (i - 1) as nat) }
            }
            List::Nil => arbitrary() // For spec purposes when out of bounds
        }
    }

    // Convert array to list
    pub fn from_array<T: Copy>(a: &[T]) -> (l: List<T>)
        ensures 
            length(&l) == a.len(),
            forall|i: int| 0 <= i < length(&l) ==> at(&l, i as nat) == a[i]
    {
        let mut l = List::Nil;
        let mut i: usize = a.len();
        
        while i > 0
            invariant 
                i <= a.len(),
                length(&l) == a.len() - i,
                forall|j: int| i <= j < a.len() ==> at(&l, (j - i as int) as nat) == a[j]
            decreases i
        {
            i = i - 1;
            l = List::Cons { head: a[i], tail: Box::new(l) };
        }
        
        l
    }
}

fn main() {
    // Example usage (recreating the original main functionality)
    let l: List<i32> = List::Cons { 
        head: 1, 
        tail: Box::new(List::Cons { 
            head: 2, 
            tail: Box::new(List::Cons { 
                head: 3, 
                tail: Box::new(List::Nil) 
            }) 
        }) 
    };
    
    let arr: [i32; 3] = [1, 2, 3];
    let _t = from_array(&arr);
}