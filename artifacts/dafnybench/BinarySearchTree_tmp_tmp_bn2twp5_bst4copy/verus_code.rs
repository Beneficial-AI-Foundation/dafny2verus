use vstd::prelude::*;

verus! {
    // Simpler BST using i32 instead of int for easier handling
    pub struct BinarySearchTree {
        pub root: Option<TreeNode>,
    }

    pub struct TreeNode {
        pub value: i32,
        pub left: Option<Box<TreeNode>>,
        pub right: Option<Box<TreeNode>>,
    }

    pub fn get_min(tree: &BinarySearchTree) -> (res: i32) {
        match &tree.root {
            None => 0,
            Some(node) => get_min_node(node)
        }
    }

    #[verifier::exec_allows_no_decreases_clause]
    pub fn get_min_node(node: &TreeNode) -> (res: i32) {
        match &node.left {
            None => node.value,
            Some(left) => get_min_node(left)
        }
    }

    pub fn get_max(tree: &BinarySearchTree) -> (res: i32) {
        match &tree.root {
            None => 0,
            Some(node) => get_max_node(node)
        }
    }

    #[verifier::exec_allows_no_decreases_clause]
    pub fn get_max_node(node: &TreeNode) -> (res: i32) {
        match &node.right {
            None => node.value,
            Some(right) => get_max_node(right)
        }
    }

    pub fn insert(tree: BinarySearchTree, value: i32) -> (res: BinarySearchTree) {
        BinarySearchTree {
            root: insert_node(tree.root, value)
        }
    }

    #[verifier::exec_allows_no_decreases_clause]
    pub fn insert_node(node: Option<TreeNode>, value: i32) -> (res: Option<TreeNode>) {
        match node {
            None => Some(TreeNode {
                value: value,
                left: None,
                right: None,
            }),
            Some(n) => {
                if value == n.value {
                    Some(n)
                } else if value < n.value {
                    let left_val = match n.left {
                        None => None,
                        Some(l) => Some(*l)
                    };
                    let new_left = insert_node(left_val, value);
                    Some(TreeNode {
                        value: n.value,
                        left: new_left.map(|x| Box::new(x)),
                        right: n.right,
                    })
                } else {
                    let right_val = match n.right {
                        None => None,
                        Some(r) => Some(*r)
                    };
                    let new_right = insert_node(right_val, value);
                    Some(TreeNode {
                        value: n.value,
                        left: n.left,
                        right: new_right.map(|x| Box::new(x)),
                    })
                }
            }
        }
    }

    pub fn delete(tree: BinarySearchTree, value: i32) -> (res: BinarySearchTree) {
        BinarySearchTree {
            root: delete_node(tree.root, value)
        }
    }

    #[verifier::exec_allows_no_decreases_clause]
    pub fn delete_node(node: Option<TreeNode>, value: i32) -> (res: Option<TreeNode>) {
        match node {
            None => None,
            Some(n) => {
                if value < n.value {
                    let left_val = match n.left {
                        None => None,
                        Some(l) => Some(*l)
                    };
                    let new_left = delete_node(left_val, value);
                    Some(TreeNode {
                        value: n.value,
                        left: new_left.map(|x| Box::new(x)),
                        right: n.right,
                    })
                } else if value > n.value {
                    let right_val = match n.right {
                        None => None,
                        Some(r) => Some(*r)
                    };
                    let new_right = delete_node(right_val, value);
                    Some(TreeNode {
                        value: n.value,
                        left: n.left,
                        right: new_right.map(|x| Box::new(x)),
                    })
                } else {
                    match (n.left, n.right) {
                        (None, None) => None,
                        (Some(l), None) => Some(*l),
                        (None, Some(r)) => Some(*r),
                        (Some(l), Some(r)) => {
                            let min_val = get_min_node(&*r);
                            let new_right = delete_node(Some(*r), min_val);
                            Some(TreeNode {
                                value: min_val,
                                left: Some(l),
                                right: new_right.map(|x| Box::new(x)),
                            })
                        }
                    }
                }
            }
        }
    }

    pub fn inorder(tree: &BinarySearchTree) {
        match &tree.root {
            None => {},
            Some(node) => inorder_node(node)
        }
    }

    #[verifier::exec_allows_no_decreases_clause]
    pub fn inorder_node(node: &TreeNode) {
        match &node.left {
            None => {},
            Some(left) => inorder_node(left)
        }
        // Would print value here
        match &node.right {
            None => {},
            Some(right) => inorder_node(right)
        }
    }

    pub fn postorder(tree: &BinarySearchTree) {
        match &tree.root {
            None => {},
            Some(node) => postorder_node(node)
        }
    }

    #[verifier::exec_allows_no_decreases_clause]
    pub fn postorder_node(node: &TreeNode) {
        match &node.left {
            None => {},
            Some(left) => postorder_node(left)
        }
        match &node.right {
            None => {},
            Some(right) => postorder_node(right)
        }
        // Would print value here
    }

    pub fn main() {
        let tree = insert(BinarySearchTree { root: None }, 3);
        let mut u = insert(tree, 2);

        u = insert(u, 7);
        u = insert(u, 6);
        u = insert(u, 9);

        inorder(&u);
        postorder(&u);

        u = delete(u, 7);
        
        inorder(&u);
        postorder(&u);
    }
}