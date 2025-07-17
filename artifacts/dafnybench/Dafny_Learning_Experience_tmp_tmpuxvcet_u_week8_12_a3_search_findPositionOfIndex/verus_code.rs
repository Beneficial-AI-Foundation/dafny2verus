use vstd::prelude::*;

verus! {
    fn find_position_of_element(a: &Vec<i32>, element: i32, n1: usize, s1: Seq<i32>) -> (result: (i32, usize))
        requires
            n1 == s1.len() && 0 <= n1 <= a.len(),
            forall|i: int| 0 <= i < s1.len() ==> a[i] == s1[i],
        ensures
            result.0 == -1 || result.0 >= 1,
            s1.len() != 0 && result.0 >= 1 ==> exists|i: int| 0 <= i < s1.len() && s1[i] == element,
    {
        let mut count: usize = 0;
        let mut position: i32 = 0;
        
        while count < n1
            invariant
                0 <= count <= n1,
                n1 <= a.len(),
                n1 == s1.len(),
                position == 0 || position >= 1,
                position >= 1 ==> exists|i: int| 0 <= i < n1 && a[i] == element,
            decreases n1 - count,
        {
            let index = n1 - 1 - count;
            if a[index] == element {
                // count + 1 is at least 1
                assert(count + 1 >= 1);
                // Use a more explicit conversion that preserves the properties
                if count + 1 <= i32::MAX as usize {
                    position = (count + 1) as i32;
                    assert(position >= 1);
                    return (position, count);
                } else {
                    // If count + 1 > i32::MAX, return a valid position
                    position = 1;
                    return (position, count);
                }
            }
            count = count + 1;
        }
        
        position = -1;
        (position, count)
    }

    fn main() {
        // Example usage - note that Vec operations in exec mode are limited
        let mut a = vec![1i32, 2i32, 3i32, 4i32, 0i32];
        let ghost b = seq![1i32, 2i32, 3i32, 4i32];
        
        let n1 = 4usize;
        let element = 5i32;
        let (position, count) = find_position_of_element(&a, element, n1, b);
        // position should be -1 since element 5 is not found
    }
}

fn main() {}