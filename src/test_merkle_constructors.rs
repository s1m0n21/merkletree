#![cfg(test)]

use crate::hash::{Algorithm, Hashable};
use crate::merkle::{
    get_merkle_proof_lemma_len, get_merkle_tree_len_generic, Element, FromIndexedParallelIterator,
    MerkleTree,
};
use crate::store::{Store, StoreConfig, VecStore};
use crate::test_common::{Item, Sha256Hasher, XOR128};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
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

fn generate_byte_slice_tree<E: Element, A: Algorithm<E>>(leaves: usize) -> Vec<u8> {
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

    dataset
}

/// Base tree constructors
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
    let dataset = generate_byte_slice_tree::<E, A>(leaves);
    MerkleTree::from_byte_slice(dataset.as_slice())
        .expect("failed to instantiate tree [from_byte_slice]")
}

fn instantiate_from_par_iter<E: Element, A: Algorithm<E>, S: Store<E>, U: Unsigned>(
    leaves: usize,
) -> MerkleTree<E, A, S, U> {
    let dataset = generate_vector_of_elements::<E>(leaves);
    MerkleTree::from_par_iter(dataset.into_par_iter())
        .expect("failed to instantiate tree [try_from_par_iter]")
}

fn instantiate_from_tree_slice<E: Element, A: Algorithm<E>, S: Store<E>, U: Unsigned>(
    leaves: usize,
) -> MerkleTree<E, A, S, U> {
    let tree = instantiate_new::<E, A, S, U>(leaves);
    let serialized_tree = serialize_tree(tree);
    MerkleTree::from_tree_slice(serialized_tree.as_slice(), leaves)
        .expect("failed to instantiate tree [from_tree_slice]")
}

fn instantiate_from_data_store<E: Element, A: Algorithm<E>, S: Store<E>, U: Unsigned>(
    leaves: usize,
) -> MerkleTree<E, A, S, U> {
    let tree = instantiate_new::<E, A, S, U>(leaves);
    let serialized_tree = serialize_tree(tree);
    let store = Store::new_from_slice(serialized_tree.len(), &serialized_tree)
        .expect("can't create new store over existing one");
    MerkleTree::from_data_store(store, leaves)
        .expect("failed to instantiate tree [from_data_store]")
}

fn instantiate_new_with_config<E: Element, A: Algorithm<E>, S: Store<E>, U: Unsigned>(
    leaves: usize,
) -> MerkleTree<E, A, S, U> {
    let dataset = generate_vector_of_elements::<E>(leaves);
    let tmp_tree = instantiate_new::<E, A, S, U>(leaves);
    let func_name = "instantiate_new_with_config";
    MerkleTree::new_with_config(
        dataset,
        get_config(leaves, tmp_tree.len(), tmp_tree.row_count(), func_name),
    )
    .expect("failed to instantiate tree [new_with_config]")
}

fn instantiate_try_from_iter_with_config<E: Element, A: Algorithm<E>, S: Store<E>, U: Unsigned>(
    leaves: usize,
) -> MerkleTree<E, A, S, U> {
    let dataset = generate_vector_of_elements::<E>(leaves);
    let tmp_tree = instantiate_try_from_iter::<E, A, S, U>(leaves);
    let func_name = "instantiate_try_from_iter_with_config";
    MerkleTree::try_from_iter_with_config(
        dataset.into_iter().map(Ok),
        get_config(leaves, tmp_tree.len(), tmp_tree.row_count(), func_name),
    )
    .expect("failed to instantiate tree [try_from_iter_with_config]")
}

fn instantiate_from_data_with_config<E: Element, A: Algorithm<E>, S: Store<E>, U: Unsigned>(
    leaves: usize,
) -> MerkleTree<E, A, S, U> {
    let dataset = generate_vector_of_usizes(leaves);
    let tmp_tree = instantiate_from_data::<E, A, S, U>(leaves);
    let func_name = "instantiate_from_data_with_config";
    MerkleTree::from_data_with_config(
        dataset.as_slice(),
        get_config(leaves, tmp_tree.len(), tmp_tree.row_count(), func_name),
    )
    .expect("failed to instantiate tree [from_data_with_config]")
}

fn instantiate_from_tree_slice_with_config<
    E: Element,
    A: Algorithm<E>,
    S: Store<E>,
    U: Unsigned,
>(
    leaves: usize,
) -> MerkleTree<E, A, S, U> {
    let tmp_tree = instantiate_new::<E, A, S, U>(leaves);
    let tree_len = tmp_tree.len();
    let row_count = tmp_tree.row_count();
    let serialized_tree = serialize_tree(tmp_tree);
    let func_name = "instantiate_from_tree_slice_with_config";
    MerkleTree::from_tree_slice_with_config(
        serialized_tree.as_slice(),
        leaves,
        get_config(leaves, tree_len, row_count, func_name),
    )
    .expect("failed to instantiate tree [from_tree_slice_with_config]")
}

fn instantiate_from_par_iter_with_config<E: Element, A: Algorithm<E>, S: Store<E>, U: Unsigned>(
    leaves: usize,
) -> MerkleTree<E, A, S, U> {
    let dataset = generate_vector_of_elements::<E>(leaves);
    let tmp_tree = instantiate_from_par_iter::<E, A, S, U>(leaves);
    let func_name = "instantiate_from_par_iter_with_config";
    MerkleTree::from_par_iter_with_config(
        dataset,
        get_config(leaves, tmp_tree.len(), tmp_tree.row_count(), func_name),
    )
    .expect("failed to instantiate tree [from_par_iter_with_config]")
}

fn instantiate_from_byte_slice_with_config<
    E: Element,
    A: Algorithm<E>,
    S: Store<E>,
    U: Unsigned,
>(
    leaves: usize,
) -> MerkleTree<E, A, S, U> {
    let dataset = generate_byte_slice_tree::<E, A>(leaves);
    let tmp_tree = instantiate_from_byte_slice::<E, A, S, U>(leaves);
    let func_name = "instantiate_from_byte_slice_with_config";
    MerkleTree::from_byte_slice_with_config(
        dataset.as_slice(),
        get_config(leaves, tmp_tree.len(), tmp_tree.row_count(), func_name),
    )
    .expect("failed to instantiate tree [from_byte_slice_with_config]")
}

/// Compound tree constructors
fn instantiate_ctree_from_trees<
    E: Element,
    A: Algorithm<E>,
    S: Store<E>,
    BaseTreeArity: Unsigned,
    SubTreeArity: Unsigned,
>(
    base_tree_constructor: fn(usize) -> MerkleTree<E, A, S, BaseTreeArity>,
    base_tree_leaves: usize,
) -> MerkleTree<E, A, S, BaseTreeArity, SubTreeArity> {
    let mut base_trees: Vec<MerkleTree<E, A, S, BaseTreeArity, U0, U0>> = Vec::new();
    for _ in 0..SubTreeArity::to_usize() {
        base_trees.push(base_tree_constructor(base_tree_leaves));
    }
    MerkleTree::from_trees(base_trees).expect("failed to instantiate compound tree [from_trees]")
}

/// Compound-compound tree constructors
fn instantiate_cctree_from_sub_trees<
    E: Element,
    A: Algorithm<E>,
    S: Store<E>,
    BaseTreeArity: Unsigned,
    SubTreeArity: Unsigned,
    TopTreeArity: Unsigned,
>(
    base_tree_constructor: fn(usize) -> MerkleTree<E, A, S, BaseTreeArity>,
    compound_tree_constructor: fn(
        fn(usize) -> MerkleTree<E, A, S, BaseTreeArity>,
        usize,
    ) -> MerkleTree<E, A, S, BaseTreeArity, SubTreeArity>,
    base_tree_leaves: usize,
) -> MerkleTree<E, A, S, BaseTreeArity, SubTreeArity, TopTreeArity> {
    let mut compound_trees: Vec<MerkleTree<E, A, S, BaseTreeArity, SubTreeArity, U0>> = Vec::new();
    for _ in 0..TopTreeArity::to_usize() {
        compound_trees.push(compound_tree_constructor(
            base_tree_constructor,
            base_tree_leaves,
        ));
    }
    MerkleTree::from_sub_trees(compound_trees)
        .expect("failed to instantiate compound-compound tree from compound trees [from_sub_trees]")
}

/// Utilities
fn serialize_tree<E: Element, A: Algorithm<E>, S: Store<E>, U: Unsigned>(
    tree: MerkleTree<E, A, S, U>,
) -> Vec<u8> {
    let data = tree.data().expect("can't get tree's data [serialize_tree]");
    let data: Vec<E> = data
        .read_range(0..data.len())
        .expect("can't read actual data [serialize_tree]");
    let mut serialized_tree = vec![0u8; E::byte_len() * data.len()];
    let mut start = 0;
    let mut end = E::byte_len();
    for element in data {
        element.copy_to_slice(&mut serialized_tree[start..end]);
        start += E::byte_len();
        end += E::byte_len();
    }
    serialized_tree
}

fn get_config(
    leaves: usize,
    tree_len: usize,
    row_count: usize,
    distinguisher: &str,
) -> StoreConfig {
    let replica = format!(
        "{}-{}-{}-{}-replica",
        distinguisher, leaves, tree_len, row_count
    );
    let temp_dir = tempdir::TempDir::new(distinguisher).unwrap();

    // we attempt to discard all intermediate layers, except bottom one (set of leaves) and top-level root of base tree
    StoreConfig::new(temp_dir.path(), replica, row_count - 2)
}

/// Actual tests
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
    base_tree_constructor: fn(usize) -> MerkleTree<E, A, S, BaseTreeArity>,
    compound_tree_constructor: fn(
        fn(usize) -> MerkleTree<E, A, S, BaseTreeArity>,
        usize,
    ) -> MerkleTree<E, A, S, BaseTreeArity, SubTreeArity>,
    base_tree_leaves: usize,
    expected_leaves: usize,
    expected_len: usize,
    expected_root: E,
) {
    let compound_tree: MerkleTree<E, A, S, BaseTreeArity, SubTreeArity, U0> =
        compound_tree_constructor(base_tree_constructor, base_tree_leaves);

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
    base_tree_constructor: fn(usize) -> MerkleTree<E, A, S, BaseTreeArity>,

    compound_tree_constructor: fn(
        fn(usize) -> MerkleTree<E, A, S, BaseTreeArity>,
        usize,
    ) -> MerkleTree<E, A, S, BaseTreeArity, SubTreeArity>,

    compound_compound_tree_constructor: fn(
        fn(usize) -> MerkleTree<E, A, S, BaseTreeArity>,
        fn(
            fn(usize) -> MerkleTree<E, A, S, BaseTreeArity>,
            usize,
        ) -> MerkleTree<E, A, S, BaseTreeArity, SubTreeArity>,
        usize,
    ) -> MerkleTree<
        E,
        A,
        S,
        BaseTreeArity,
        SubTreeArity,
        TopTreeArity,
    >,

    base_tree_leaves: usize,
    expected_leaves: usize,
    expected_len: usize,
    expected_root: E,
) {
    let compound_compound_tree: MerkleTree<E, A, S, BaseTreeArity, SubTreeArity, TopTreeArity> =
        compound_compound_tree_constructor(
            base_tree_constructor,
            compound_tree_constructor,
            base_tree_leaves,
        );

    test_tree_functionality::<E, A, S, BaseTreeArity, SubTreeArity, TopTreeArity>(
        compound_compound_tree,
        expected_leaves,
        expected_len,
        expected_root,
    );
}

#[test]
fn test_from_tree_slice_group() {
    let base_tree_leaves = 4;
    let expected_total_leaves = base_tree_leaves;
    let root = Item::from_slice(&[29, 0, 28, 0, 4, 0, 4, 0, 12, 0, 12, 0, 4, 0, 4, 0]);
    let len = get_merkle_tree_len_generic::<U2, U0, U0>(base_tree_leaves).unwrap();
    let from_tree_slice = instantiate_from_tree_slice;
    run_test_base_tree::<Item, XOR128, VecStore<Item>, U2>(
        from_tree_slice,
        base_tree_leaves,
        expected_total_leaves,
        len,
        root,
    );
    let from_tree_slice_with_config = instantiate_from_tree_slice_with_config;
    run_test_base_tree::<Item, XOR128, VecStore<Item>, U2>(
        from_tree_slice_with_config,
        base_tree_leaves,
        expected_total_leaves,
        len,
        root,
    );

    let from_data_store = instantiate_from_data_store;
    run_test_base_tree::<Item, XOR128, VecStore<Item>, U2>(
        from_data_store,
        base_tree_leaves,
        expected_total_leaves,
        len,
        root,
    );

    let expected_total_leaves = base_tree_leaves * 3;
    let root = Item::from_slice(&[1, 29, 0, 28, 0, 4, 0, 4, 0, 12, 0, 12, 0, 4, 0, 4]);
    let len = get_merkle_tree_len_generic::<U2, U3, U0>(base_tree_leaves).unwrap();
    let from_trees = instantiate_ctree_from_trees;
    run_test_compound_tree::<Item, XOR128, VecStore<Item>, U2, U3>(
        from_tree_slice,
        from_trees,
        base_tree_leaves,
        expected_total_leaves,
        len,
        root,
    );

    let expected_total_leaves = base_tree_leaves * 3 * 5;
    let root = Item::from_slice(&[5, 1, 29, 0, 28, 0, 4, 0, 4, 0, 12, 0, 12, 0, 4, 0]);
    let len = get_merkle_tree_len_generic::<U2, U3, U5>(base_tree_leaves).unwrap();
    let from_sub_trees = instantiate_cctree_from_sub_trees;
    run_test_compound_compound_tree::<Item, XOR128, VecStore<Item>, U2, U3, U5>(
        from_tree_slice,
        from_trees,
        from_sub_trees,
        base_tree_leaves,
        expected_total_leaves,
        len,
        root,
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
    let from_data_with_config = instantiate_from_data_with_config;
    run_test_base_tree::<Item, XOR128, VecStore<Item>, U2>(
        from_data_with_config,
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
    let from_byte_slice_with_config = instantiate_from_byte_slice_with_config;
    run_test_base_tree::<Item, XOR128, VecStore<Item>, U2>(
        from_byte_slice_with_config,
        base_tree_leaves,
        expected_total_leaves,
        len,
        root,
    );

    let expected_total_leaves = base_tree_leaves * 3;
    let root = Item::from_slice(&[1, 1, 0, 0, 240, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    let len = get_merkle_tree_len_generic::<U2, U3, U0>(base_tree_leaves).unwrap();
    let from_trees = instantiate_ctree_from_trees;
    run_test_compound_tree::<Item, XOR128, VecStore<Item>, U2, U3>(
        from_data,
        from_trees,
        base_tree_leaves,
        expected_total_leaves,
        len,
        root,
    );

    let expected_total_leaves = base_tree_leaves * 3 * 5;
    let root = Item::from_slice(&[1, 1, 1, 0, 0, 240, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    let len = get_merkle_tree_len_generic::<U2, U3, U5>(base_tree_leaves).unwrap();
    let from_sub_trees = instantiate_cctree_from_sub_trees;
    run_test_compound_compound_tree::<Item, XOR128, VecStore<Item>, U2, U3, U5>(
        from_data,
        from_trees,
        from_sub_trees,
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
    let try_from_iter_with_config = instantiate_try_from_iter_with_config;
    run_test_base_tree::<Item, XOR128, VecStore<Item>, U2>(
        try_from_iter_with_config,
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
    let new_with_config = instantiate_new_with_config;
    run_test_base_tree::<Item, XOR128, VecStore<Item>, U2>(
        new_with_config,
        base_tree_leaves,
        expected_total_leaves,
        len,
        root,
    );
    let from_par_iter = instantiate_from_par_iter;
    run_test_base_tree::<Item, XOR128, VecStore<Item>, U2>(
        from_par_iter,
        base_tree_leaves,
        expected_total_leaves,
        len,
        root,
    );
    let from_par_iter_with_config = instantiate_from_par_iter_with_config;
    run_test_base_tree::<Item, XOR128, VecStore<Item>, U2>(
        from_par_iter_with_config,
        base_tree_leaves,
        expected_total_leaves,
        len,
        root,
    );

    let expected_total_leaves = base_tree_leaves * 3;
    let root = Item::from_slice(&[1, 29, 0, 28, 0, 4, 0, 4, 0, 12, 0, 12, 0, 4, 0, 4]);
    let len = get_merkle_tree_len_generic::<U2, U3, U0>(base_tree_leaves).unwrap();
    let from_trees = instantiate_ctree_from_trees;
    run_test_compound_tree::<Item, XOR128, VecStore<Item>, U2, U3>(
        try_from_iter,
        from_trees,
        base_tree_leaves,
        expected_total_leaves,
        len,
        root,
    );

    let expected_total_leaves = base_tree_leaves * 3 * 5;
    let root = Item::from_slice(&[5, 1, 29, 0, 28, 0, 4, 0, 4, 0, 12, 0, 12, 0, 4, 0]);
    let len = get_merkle_tree_len_generic::<U2, U3, U5>(base_tree_leaves).unwrap();
    let from_sub_trees = instantiate_cctree_from_sub_trees;
    run_test_compound_compound_tree::<Item, XOR128, VecStore<Item>, U2, U3, U5>(
        try_from_iter,
        from_trees,
        from_sub_trees,
        base_tree_leaves,
        expected_total_leaves,
        len,
        root,
    );
}
