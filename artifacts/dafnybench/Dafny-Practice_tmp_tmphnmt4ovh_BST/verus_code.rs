use vstd::prelude::*;

verus! {
    #[derive(PartialEq, Eq)]
    pub enum Tree {
        Empty,
        Node(int, Box<Tree>, Box<Tree>),
    }

    pub fn main() {
        // Simple test case
        let empty_tree = Tree::Empty;
        print_tree_numbers_inorder(&empty_tree);
    }

    pub fn print_tree_numbers_inorder(t: &Tree)
        decreases t
    {
        match t {
            Tree::Empty => {},
            Tree::Node(n, l, r) => {
                print_tree_numbers_inorder(l);
                // print! would be used here in actual code
                print_tree_numbers_inorder(r);
            }
        }
    }

    pub open spec fn numbers_in_tree(t: Tree) -> Set<int> {
        numbers_in_sequence(inorder(t))
    }

    pub open spec fn numbers_in_sequence(q: Seq<int>) -> Set<int> {
        Set::new(|x: int| q.contains(x))
    }

    pub open spec fn bst(t: Tree) -> bool {
        ascending(inorder(t))
    }

    pub open spec fn inorder(t: Tree) -> Seq<int>
        decreases t
    {
        match t {
            Tree::Empty => Seq::empty(),
            Tree::Node(n, nt1, nt2) => inorder(*nt1) + seq![n] + inorder(*nt2),
        }
    }

    pub open spec fn ascending(q: Seq<int>) -> bool {
        forall|i: int, j: int| 0 <= i < j < q.len() ==> q[i] < q[j]
    }

    pub open spec fn no_duplicates(q: Seq<int>) -> bool {
        forall|i: int, j: int| 0 <= i < j < q.len() ==> q[i] != q[j]
    }

    pub fn build_bst(q: Vec<int>) -> (t: Tree)
        requires no_duplicates(q@),
        ensures bst(t) && numbers_in_tree(t) == numbers_in_sequence(q@)
    {
        let mut t = Tree::Empty;
        let mut i = 0;
        while i < q.len()
            invariant 
                bst(t),
                numbers_in_tree(t) == numbers_in_sequence(q@.subrange(0, i as int)),
                0 <= i <= q.len()
            decreases q.len() - i
        {
            t = insert_bst(t, q[i]);
            i += 1;
        }
        t
    }

    pub fn insert_bst(t0: Tree, x: int) -> (t: Tree)
        requires bst(t0) && !numbers_in_tree(t0).contains(x),
        ensures bst(t) && numbers_in_tree(t) == numbers_in_tree(t0).insert(x),
        decreases t0
    {
        match t0 {
            Tree::Empty => Tree::Node(x, Box::new(Tree::Empty), Box::new(Tree::Empty)),
            Tree::Node(i, left, right) => {
                if x < i {
                    proof {
                        lemma_binary_search_subtree(i, *left, *right);
                    }
                    let tmp = insert_bst(*left, x);
                    let t = Tree::Node(i, Box::new(tmp), right);
                    
                    // Ghost variables and assertions
                    proof {
                        let right_nums = inorder(*right);
                        let left_nums = inorder(*left);
                        let all_nums = inorder(t0);
                        assert(all_nums == left_nums + seq![i] + right_nums);
                        assert(all_nums[left_nums.len() as int] == i);
                        assert(ascending(right_nums));
                        assert(ascending(left_nums));
                        assert(ascending(left_nums + seq![i] + right_nums));
                        
                        let new_all_nums = inorder(t);
                        let new_left_nums = inorder(tmp);
                        assert(new_all_nums == (new_left_nums + seq![i] + right_nums));
                        assert(ascending(seq![i] + right_nums));
                        assert(ascending(new_left_nums));
                        assert(numbers_in_sequence(new_left_nums) == numbers_in_sequence(left_nums).insert(x));
                        
                        assert(x < i);
                        lemma_all_small(new_left_nums, i);
                        assert(forall|j: int| 0 <= j < new_left_nums.len() ==> new_left_nums[j] < i);
                        assert(ascending(new_left_nums + seq![i]));
                        assert(ascending(inorder(t)));
                        assert(bst(t));
                    }
                    
                    t
                } else {
                    proof {
                        lemma_binary_search_subtree(i, *left, *right);
                    }
                    let tmp = insert_bst(*right, x);
                    let t = Tree::Node(i, left, Box::new(tmp));
                    
                    // Ghost variables and assertions
                    proof {
                        let right_nums = inorder(*right);
                        let left_nums = inorder(*left);
                        let all_nums = inorder(t0);
                        assert(all_nums == left_nums + seq![i] + right_nums);
                        assert(all_nums[left_nums.len() as int] == i);
                        assert(ascending(right_nums));
                        assert(ascending(left_nums));
                        assert(ascending(left_nums + seq![i] + right_nums));
                        
                        let new_all_nums = inorder(t);
                        let new_right_nums = inorder(tmp);
                        assert(new_all_nums == (left_nums + seq![i] + new_right_nums));
                        assert(ascending(left_nums + seq![i]));
                        assert(ascending(new_right_nums));
                        assert(numbers_in_sequence(new_right_nums) == numbers_in_sequence(right_nums).insert(x));
                        
                        assert(x > i);
                        lemma_all_big(new_right_nums, i);
                        assert(forall|j: int| 0 <= j < new_right_nums.len() ==> new_right_nums[j] > i);
                        assert(ascending(inorder(t)));
                        assert(bst(t));
                    }
                    
                    t
                }
            }
        }
    }

    pub proof fn lemma_binary_search_subtree(n: int, left: Tree, right: Tree)
        requires bst(Tree::Node(n, Box::new(left), Box::new(right))),
        ensures bst(left) && bst(right)
    {
        assert(ascending(inorder(Tree::Node(n, Box::new(left), Box::new(right)))));
        let qleft = inorder(left);
        let qright = inorder(right);
        let q = qleft + seq![n] + qright;
        assert(q == inorder(Tree::Node(n, Box::new(left), Box::new(right))));
        assert(ascending(qleft + seq![n] + qright));
        lemma_ascending_subsequence(q, qleft, 0);
        assert(ascending(qleft));
        lemma_ascending_subsequence(q, qright, qleft.len() as int + 1);
        assert(ascending(qright));
    }

    pub proof fn lemma_ascending_subsequence(q1: Seq<int>, q2: Seq<int>, i: int)
        requires 
            i <= q1.len() - q2.len() && 
            q2 == q1.subrange(i, i + q2.len()) &&
            ascending(q1),
        ensures ascending(q2)
    {}

    pub proof fn lemma_all_small(q: Seq<int>, i: int)
        requires 
            forall|k: int| numbers_in_sequence(q).contains(k) ==> k < i,
            forall|k: int| 0 <= k < q.len() ==> numbers_in_sequence(q).contains(q[k]),
        ensures forall|j: int| 0 <= j < q.len() ==> q[j] < i
    {}

    pub proof fn lemma_all_big(q: Seq<int>, i: int)
        requires 
            forall|k: int| numbers_in_sequence(q).contains(k) ==> k > i,
            forall|k: int| 0 <= k < q.len() ==> numbers_in_sequence(q).contains(q[k]),
        ensures forall|j: int| 0 <= j < q.len() ==> q[j] > i
    {}
}