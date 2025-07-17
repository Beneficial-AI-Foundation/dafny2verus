use vstd::prelude::*;

verus! {
    /**
      Inverts an array of ints.
     */
    fn invert_array(a: &mut Vec<int>)
        requires old(a).len() > 0,
        ensures 
            a.len() == old(a).len(),
            forall|i: int| 0 <= i < a.len() ==> a[i] == old(a)[a.len() - 1 - i],
    {
        let mut index = 0;
        
        while index < a.len() / 2
            invariant 
                0 <= index <= a.len() / 2,
                a.len() == old(a).len(),
                // the elements i before position index are already switched with the old value of position a.len() - 1 - i
                forall|i: int| 0 <= i < index ==> a[i] == old(a)[a.len() - 1 - i],
                // the elements of form a.len() - 1 - i after position a.len() - 1 - index are already switched with the old value of position i
                forall|i: int| 0 <= i < index ==> a[a.len() - 1 - i] == old(a)[i],
                // the elements between index and a.len() - index are unchanged : the middle of the array
                forall|i: int| index <= i < a.len() - index ==> a[i] == old(a)[i],
            decreases a.len() / 2 - index,
        {
            let temp = a[index];
            let other_index = a.len() - 1 - index;
            let other_val = a[other_index];
            a.set(index, other_val);
            a.set(other_index, temp);
            index = index + 1;
        }
    }
}