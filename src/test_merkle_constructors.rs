#![cfg(test)]

use crate::hash::{Algorithm, Hashable};
use crate::merkle::{get_merkle_tree_len_generic, Element, MerkleTree};
use crate::store::{Store, VecStore};
use crate::test_common::{Item, Sha256Hasher, XOR128};
use typenum::{Unsigned, U0, U2, U3, U5};

/// Dataset generators. It is assumed that every generator will produce dataset with particular length, equal to leaves parameter
fn generate_vector_of_elements<E: Element>(leaves: usize) -> Vec<E> {
    let mut dataset = Vec::<E>::new();
    for index in 0..leaves {
        // we are ok with usize -> u8 conversion problems, since we need just predictable dataset
        let vector: Vec<u8> = (0..E::byte_len()).map(|x| (index + x) as u8).collect();
        let element = E::from_slice(vector.as_slice());
        dataset.push(element);
    }
    dataset
}
fn generate_vector_of_usizes(leaves: usize) -> Vec<usize> {
    let mut dataset = Vec::with_capacity(leaves);
    for i in 0..leaves {
        dataset.push(i * 93);
    }
    dataset
}

/// Constructors
fn instantiate_from_data<E: Element, A: Algorithm<E>, S: Store<E>, U: Unsigned>(
    leaves: usize,
) -> MerkleTree<E, A, S, U> {
    let dataset = generate_vector_of_usizes(leaves);
    MerkleTree::from_data(dataset.as_slice()).expect("failed to instantiate tree [from_data]")
}

fn instantiate_try_from_iter<E: Element, A: Algorithm<E>, S: Store<E>, U: Unsigned>(
    leaves: usize,
) -> MerkleTree<E, A, S, U> {
    let dataset = generate_vector_of_elements::<E>(leaves);
    MerkleTree::try_from_iter(dataset.into_iter().map(Ok))
        .expect("failed to instantiate tree [try_from_iter]")
}

fn instantiate_new<E: Element, A: Algorithm<E>, S: Store<E>, U: Unsigned>(
    leaves: usize,
) -> MerkleTree<E, A, S, U> {
    let dataset = generate_vector_of_elements::<E>(leaves);
    MerkleTree::new(dataset).expect("failed to instantiate tree [new]")
}

fn instantiate_from_byte_slice<E: Element, A: Algorithm<E>, S: Store<E>, U: Unsigned>(
    leaves: usize,
) -> MerkleTree<E, A, S, U> {
    let mut a = A::default();
    let mut a2 = A::default();

    let dataset: Vec<u8> = generate_vector_of_usizes(leaves)
        .iter()
        .map(|x| {
            a.reset();
            x.hash(&mut a);
            a.hash()
        })
        .take(leaves)
        .map(|item| {
            a2.reset();
            a2.leaf(item).as_ref().to_vec()
        })
        .flatten()
        .collect();

    MerkleTree::from_byte_slice(dataset.as_slice())
        .expect("failed to instantiate tree [from_byte_slice]")
}

fn test_tree_functionality<
    E: Element,
    A: Algorithm<E>,
    S: Store<E>,
    BaseTreeArity: Unsigned,
    SubTreeArity: Unsigned,
    TopTreeArity: Unsigned,
>(
    tree: MerkleTree<E, A, S, BaseTreeArity, SubTreeArity, TopTreeArity>,
    expected_leaves: usize,
    expected_len: usize,
    expected_root: E,
) {
    assert_eq!(tree.leafs(), expected_leaves);
    assert_eq!(tree.len(), expected_len);
    assert_eq!(tree.root(), expected_root);

    for index in 0..tree.leafs() {
        let p = tree.gen_proof(index).unwrap();
        assert!(p.validate::<A>().expect("failed to validate"));
    }
}

// base tree has SubTreeArity and TopTreeArity parameters equal to zero
fn run_test_base_tree<E: Element, A: Algorithm<E>, S: Store<E>, BaseTreeArity: Unsigned>(
    constructor: fn(usize) -> MerkleTree<E, A, S, BaseTreeArity>,
    leaves_in_tree: usize,
    expected_leaves: usize,
    expected_len: usize,
    expected_root: E,
) {
    let tree: MerkleTree<E, A, S, BaseTreeArity, U0, U0> = constructor(leaves_in_tree);
    test_tree_functionality::<E, A, S, BaseTreeArity, U0, U0>(
        tree,
        expected_leaves,
        expected_len,
        expected_root,
    );
}

// compound tree has TopTreeArity parameter equals to zero
fn run_test_compound_tree<
    E: Element,
    A: Algorithm<E>,
    S: Store<E>,
    BaseTreeArity: Unsigned,
    SubTreeArity: Unsigned,
>(
    constructor: fn(usize) -> MerkleTree<E, A, S, BaseTreeArity>,
    base_tree_leaves: usize,
    expected_leaves: usize,
    expected_len: usize,
    expected_root: E,
) {
    let mut base_trees: Vec<MerkleTree<E, A, S, BaseTreeArity, U0, U0>> = Vec::new();
    for _ in 0..SubTreeArity::to_usize() {
        base_trees.push(constructor(base_tree_leaves));
    }

    let compound_tree: MerkleTree<E, A, S, BaseTreeArity, SubTreeArity, U0> =
        MerkleTree::from_trees(base_trees)
            .expect("failed to create compound tree from base trees with vector storage");

    test_tree_functionality::<E, A, S, BaseTreeArity, SubTreeArity, U0>(
        compound_tree,
        expected_leaves,
        expected_len,
        expected_root,
    );
}

// compound-compound tree has all non-zero arities
fn run_test_compound_compound_tree<
    E: Element,
    A: Algorithm<E>,
    S: Store<E>,
    BaseTreeArity: Unsigned,
    SubTreeArity: Unsigned,
    TopTreeArity: Unsigned,
>(
    constructor: fn(usize) -> MerkleTree<E, A, S, BaseTreeArity>,
    base_tree_leaves: usize,
    expected_leaves: usize,
    expected_len: usize,
    expected_root: E,
) {
    let mut sub_trees: Vec<MerkleTree<E, A, S, BaseTreeArity, SubTreeArity, U0>> = Vec::new();
    for _ in 0..TopTreeArity::to_usize() {
        let mut base_trees: Vec<MerkleTree<E, A, S, BaseTreeArity, U0, U0>> = Vec::new();
        for _ in 0..SubTreeArity::to_usize() {
            base_trees.push(constructor(base_tree_leaves));
        }
        let sub_tree: MerkleTree<E, A, S, BaseTreeArity, SubTreeArity, U0> =
            MerkleTree::from_trees(base_trees)
                .expect("failed to create compound tree from base trees with vector storage");
        sub_trees.push(sub_tree);
    }

    let compound_compound_tree: MerkleTree<E, A, S, BaseTreeArity, SubTreeArity, TopTreeArity> =
        MerkleTree::from_sub_trees(sub_trees).expect(
            "failed to create compound-compound tree from compound trees with vector storage",
        );

    test_tree_functionality::<E, A, S, BaseTreeArity, SubTreeArity, TopTreeArity>(
        compound_compound_tree,
        expected_leaves,
        expected_len,
        expected_root,
    );
}

#[test]
fn test_from_data_group() {
    let base_tree_leaves = 4;
    let expected_total_leaves = base_tree_leaves;
    let root = Item::from_slice(&[1, 0, 0, 240, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    let len = get_merkle_tree_len_generic::<U2, U0, U0>(base_tree_leaves).unwrap();
    let from_data = instantiate_from_data;
    run_test_base_tree::<Item, XOR128, VecStore<Item>, U2>(
        from_data,
        base_tree_leaves,
        expected_total_leaves,
        len,
        root,
    );
    let from_byte_slice = instantiate_from_byte_slice;
    run_test_base_tree::<Item, XOR128, VecStore<Item>, U2>(
        from_byte_slice,
        base_tree_leaves,
        expected_total_leaves,
        len,
        root,
    );

    let expected_total_leaves = base_tree_leaves * 3;
    let root = Item::from_slice(&[1, 1, 0, 0, 240, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    let len = get_merkle_tree_len_generic::<U2, U3, U0>(base_tree_leaves).unwrap();
    run_test_compound_tree::<Item, XOR128, VecStore<Item>, U2, U3>(
        from_data,
        base_tree_leaves,
        expected_total_leaves,
        len,
        root,
    );
    run_test_compound_tree::<Item, XOR128, VecStore<Item>, U2, U3>(
        from_byte_slice,
        base_tree_leaves,
        expected_total_leaves,
        len,
        root,
    );

    let expected_total_leaves = base_tree_leaves * 3 * 5;
    let root = Item::from_slice(&[1, 1, 1, 0, 0, 240, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    let len = get_merkle_tree_len_generic::<U2, U3, U5>(base_tree_leaves).unwrap();
    run_test_compound_compound_tree::<Item, XOR128, VecStore<Item>, U2, U3, U5>(
        from_data,
        base_tree_leaves,
        expected_total_leaves,
        len,
        root,
    );
    run_test_compound_compound_tree::<Item, XOR128, VecStore<Item>, U2, U3, U5>(
        from_byte_slice,
        base_tree_leaves,
        expected_total_leaves,
        len,
        root,
    );
}

#[test]
fn test_try_from_iter_group() {
    let base_tree_leaves = 4;
    let expected_total_leaves = base_tree_leaves;
    let root = Item::from_slice(&[29, 0, 28, 0, 4, 0, 4, 0, 12, 0, 12, 0, 4, 0, 4, 0]);
    let len = get_merkle_tree_len_generic::<U2, U0, U0>(base_tree_leaves).unwrap();
    let try_from_iter = instantiate_try_from_iter;
    run_test_base_tree::<Item, XOR128, VecStore<Item>, U2>(
        try_from_iter,
        base_tree_leaves,
        expected_total_leaves,
        len,
        root,
    );
    let new = instantiate_new;
    run_test_base_tree::<Item, XOR128, VecStore<Item>, U2>(
        new,
        base_tree_leaves,
        expected_total_leaves,
        len,
        root,
    );

    let expected_total_leaves = base_tree_leaves * 3;
    let root = Item::from_slice(&[1, 29, 0, 28, 0, 4, 0, 4, 0, 12, 0, 12, 0, 4, 0, 4]);
    let len = get_merkle_tree_len_generic::<U2, U3, U0>(base_tree_leaves).unwrap();
    run_test_compound_tree::<Item, XOR128, VecStore<Item>, U2, U3>(
        try_from_iter,
        base_tree_leaves,
        expected_total_leaves,
        len,
        root,
    );
    run_test_compound_tree::<Item, XOR128, VecStore<Item>, U2, U3>(
        new,
        base_tree_leaves,
        expected_total_leaves,
        len,
        root,
    );

    let expected_total_leaves = base_tree_leaves * 3 * 5;
    let root = Item::from_slice(&[5, 1, 29, 0, 28, 0, 4, 0, 4, 0, 12, 0, 12, 0, 4, 0]);
    let len = get_merkle_tree_len_generic::<U2, U3, U5>(base_tree_leaves).unwrap();
    run_test_compound_compound_tree::<Item, XOR128, VecStore<Item>, U2, U3, U5>(
        try_from_iter,
        base_tree_leaves,
        expected_total_leaves,
        len,
        root,
    );
    run_test_compound_compound_tree::<Item, XOR128, VecStore<Item>, U2, U3, U5>(
        new,
        base_tree_leaves,
        expected_total_leaves,
        len,
        root,
    );
}
