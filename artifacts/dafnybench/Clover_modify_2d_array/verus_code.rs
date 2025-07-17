use vstd::prelude::*;

verus! {
    // Translation of the original Dafny method to Verus
    // Since Verus sequences are immutable, this is a spec function that returns a new sequence
    spec fn modify_array_element(arr: Seq<Seq<nat>>, index1: int, index2: int, val: nat) -> Seq<Seq<nat>>
        recommends
            0 <= index1 < arr.len(),
            0 <= index2 < arr.index(index1).len(),
            // All inner arrays are distinct (corresponding to Dafny's requirement)
            forall|i: int, j: int| 
                0 <= i < arr.len() && 0 <= j < arr.len() && i != j ==> 
                arr.index(i) != arr.index(j),
    {
        arr.update(index1, arr.index(index1).update(index2, val))
    }

    // Proof function that verifies the postconditions from the original Dafny code
    proof fn modify_array_element_properties(arr: Seq<Seq<nat>>, index1: int, index2: int, val: nat)
        requires
            0 <= index1 < arr.len(),
            0 <= index2 < arr.index(index1).len(),
            // All inner arrays are distinct
            forall|i: int, j: int| 
                0 <= i < arr.len() && 0 <= j < arr.len() && i != j ==> 
                arr.index(i) != arr.index(j),
        ensures ({
            let result = modify_array_element(arr, index1, index2, val);
            // Array structure is preserved (corresponds to Dafny's "arr[i] == old(arr[i])")
            &&& result.len() == arr.len()
            &&& forall|i: int| 0 <= i < result.len() ==> 
                if i == index1 { 
                    result.index(i).len() == arr.index(i).len()
                } else { 
                    result.index(i) == arr.index(i) 
                }
            // All elements remain the same except the modified one
            &&& forall|i: int, j: int| 
                0 <= i < result.len() && 0 <= j < result.index(i).len() && 
                (i != index1 || j != index2) ==> 
                result.index(i).index(j) == arr.index(i).index(j)
            // The modified element has the new value
            &&& result.index(index1).index(index2) == val
        })
    {
        // The proof is automatic for this simple case
    }
}