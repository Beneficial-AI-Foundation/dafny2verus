use vstd::prelude::*;

verus! {
    // Ghost predicate to check if str2 exists as a substring in str1
    spec fn exists_substring(str1: Seq<char>, str2: Seq<char>) -> bool {
        exists|offset: int| 0 <= offset <= str1.len() && 
            str2.len() <= str1.len() - offset &&
            str2 =~= str1.subrange(offset, offset + str2.len() as int)
    }

    // Postcondition predicate
    spec fn post(str1: Seq<char>, str2: Seq<char>, found: bool, i: nat) -> bool {
        (found <==> exists_substring(str1, str2)) &&
        (found ==> i + str2.len() <= str1.len() && 
            str2 =~= str1.subrange(i as int, i as int + str2.len() as int))
    }

    // Outer loop invariant
    spec fn outter_inv_correctness(str1: Seq<char>, str2: Seq<char>, found: bool, i: nat) -> bool {
        (found ==> (i + str2.len() <= str1.len() && 
            str2 =~= str1.subrange(i as int, i as int + str2.len() as int))) &&
        (!found && 0 < i <= str1.len() && i != str2.len() - 1 ==> 
            !exists_substring(str1.subrange(0, i as int), str2)) &&
        (!found ==> i <= str1.len())
    }

    // Inner loop invariant for correctness
    spec fn inner_inv_correctness(str1: Seq<char>, str2: Seq<char>, i: nat, j: int, found: bool) -> bool {
        0 <= j <= i &&
        j < str2.len() &&
        i < str1.len() &&
        (str1.index(i as int) == str2.index(j) ==> 
            str2.subrange(j, str2.len() as int) =~= str1.subrange(i as int, i as int + str2.len() as int - j)) &&
        (found ==> j == 0 && str1.index(i as int) == str2.index(j))
    }

    // Inner loop invariant for termination
    spec fn inner_inv_termination(str1: Seq<char>, str2: Seq<char>, i: nat, j: int, old_i: nat, old_j: int) -> bool {
        old_j - j == old_i - i
    }

    // Lemma 1 from the original code
    proof fn lemma1(str1: Seq<char>, str2: Seq<char>, i: nat, j: int, old_i: nat, old_j: int, found: bool)
        requires
            !found,
            str2.len() > 0,
            outter_inv_correctness(str1, str2, found, old_i),
            i + str2.len() - j == old_i + 1,
            old_i + 1 >= str2.len(),
            old_i + 1 <= str1.len(),
            0 <= i < str1.len() && 0 <= j < str2.len(),
            str1.index(i as int) != str2.index(j),
            str2.len() > 0,
            (0 < old_i <= str1.len() ==> !exists_substring(str1.subrange(0, old_i as int), str2)),
        ensures
            (0 < old_i + 1 <= str1.len() ==> !exists_substring(str1.subrange(0, old_i as int + 1), str2)),
    {
        // Proof body would be filled in with the logic from the original lemma
    }

    // The main function translated from Dafny method
    fn find_first_occurrence(str1: Seq<char>, str2: Seq<char>) -> (result: (bool, nat))
        ensures post(str1, str2, result.0, result.1)
    {
        if str2.len() == 0 {
            let found = true;
            let i = 0nat;
            proof { assert(post(str1, str2, found, i)); }
            (found, i)
        } else if str1.len() < str2.len() {
            let found = false;
            let i = 0nat;
            proof { assert(post(str1, str2, found, i)); }
            (found, i)
        } else {
            let mut found = false;
            let mut i = str2.len() - 1;
            
            proof {
                assert(str2.len() > 0);
                assert(str1.len() >= str2.len());
                assert(outter_inv_correctness(str1, str2, false, str2.len() - 1));
            }
            
            while !found && i < str1.len()
                invariant 
                    outter_inv_correctness(str1, str2, found, i),
                decreases if !found { 1nat } else { 0nat }, str1.len() - i,
            {
                proof {
                    assert(outter_inv_correctness(str1, str2, found, i));
                    assert(str2.len() > 0);
                    assert(!found && i < str1.len());
                }
                
                let mut j;
                let ghost old_i = i;
                let ghost old_j;
                
                proof {
                    j = (str2.len() - 1) as int;
                    old_j = j;
                }
                
                while !found && {
                    proof {
                        j >= 0 && i < str1.len() && j < str2.len() && str1.index(i as int) == str2.index(j)
                    }
                }
                    invariant 
                        inner_inv_termination(str1, str2, i, j, old_i, old_j),
                        inner_inv_correctness(str1, str2, i, j, found),
                    decreases j, if !found { 1nat } else { 0nat },
                {
                    proof {
                        if j == 0 {
                            assert(j == 0 && str1.index(i as int) == str2.index(j));
                            found = true;
                            assert(inner_inv_termination(str1, str2, i, j, old_i, old_j));
                            assert(inner_inv_correctness(str1, str2, i, j, found));
                        } else {
                            assert(j > 0);
                            assert(inner_inv_termination(str1, str2, i - 1, j - 1, old_i, old_j));
                            assert(inner_inv_correctness(str1, str2, i - 1, j - 1, found));
                            i = i - 1;
                            j = j - 1;
                            assert(inner_inv_termination(str1, str2, i, j, old_i, old_j));
                            assert(inner_inv_correctness(str1, str2, i, j, found));
                        }
                        assert(j >= 0);
                        assert(inner_inv_termination(str1, str2, i, j, old_i, old_j));
                        assert(inner_inv_correctness(str1, str2, i, j, found));
                    }
                }
                
                if !found {
                    proof {
                        assert(i < str1.len());
                        assert(str2.len() > 0);
                        assert(old_j - j == old_i - i);
                        assert(old_i < i + str2.len() - j);
                        assert(outter_inv_correctness(str1, str2, found, old_i));
                        assert(i + str2.len() - j == old_i + 1);
                        assert(str1.index(i as int) != str2.index(j));
                    }
                    
                    lemma1(str1, str2, i, j, old_i, old_j, found);
                    
                    proof {
                        assert(0 < old_i + 1 <= str1.len() ==> !exists_substring(str1.subrange(0, old_i as int + 1), str2));
                        assert(0 < i + str2.len() - j <= str1.len() ==> !exists_substring(str1.subrange(0, i as int + str2.len() - j), str2));
                        assert(outter_inv_correctness(str1, str2, found, i + str2.len() - j));
                    }
                    
                    i = i + str2.len() - j;
                    
                    proof {
                        assert(old_i < i);
                        assert(outter_inv_correctness(str1, str2, found, i));
                        assert(i <= str1.len());
                    }
                }
                
                proof {
                    assert(!found ==> i <= str1.len());
                    assert(!found ==> old_i < i);
                    assert(!found ==> outter_inv_correctness(str1, str2, found, i));
                    assert(found ==> outter_inv_correctness(str1, str2, found, i));
                    assert(outter_inv_correctness(str1, str2, found, i));
                }
            }
            
            proof {
                assert(outter_inv_correctness(str1, str2, found, i));
                assert(found ==> i + str2.len() <= str1.len() && str2 =~= str1.subrange(i as int, i as int + str2.len() as int));
                assert(!found && 0 < i <= str1.len() ==> !exists_substring(str1.subrange(0, i as int), str2));
                assert(!found ==> i <= str1.len());
                assert(found || i >= str1.len());
                assert(!found && i == str1.len() ==> !exists_substring(str1.subrange(0, i as int), str2));
                assert(i == str1.len() ==> str1.subrange(0, i as int) == str1);
                assert(!found && i == str1.len() ==> !exists_substring(str1, str2));
                assert(!found ==> i >= str1.len());
                assert(!found ==> i == str1.len());
                assert(!found ==> !exists_substring(str1, str2));
                assert(found ==> exists_substring(str1, str2));
                assert(found <==> exists_substring(str1, str2));
                assert(found ==> i + str2.len() <= str1.len() && str2 =~= str1.subrange(i as int, i as int + str2.len() as int));
                assert(post(str1, str2, found, i));
            }
            
            (found, i)
        }
    }

    fn main() {
        let str1_a = seq!['s', 't', 'r', 'i', 'n', 'g'];
        let str1_b = seq![' ', 'i', 'n', ' ', 'D', 'a', 'f', 'n', 'y'];
        let str1 = str1_a.add(str1_b);
        let str2 = seq!['r', 'i', 'n', 'g'];
        let (found, i) = find_first_occurrence(str1, str2);
        
        proof {
            assert(found) by {
                assert(exists_substring(str1, str2)) by {
                    let offset = 2int;
                    assert(0 <= offset <= str1.len());
                    assert(str2 =~= str1.subrange(offset, offset + str2.len() as int));
                }
            }
        }
        
        println!("Found: {}, Index: {}", found, i);
    }
}