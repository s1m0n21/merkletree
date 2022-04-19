mod common;

use rayon::iter::IntoParallelIterator;

use crate::common::{
    dump_tree_data_to_replica, generate_byte_slice_tree, generate_vector_of_elements,
    generate_vector_of_usizes, serialize_tree, test_levelcache_tree_functionality, TestItem,
    TestItemType, TestSha256Hasher, TestXOR128,
};
use merkletree::hash::Algorithm;
use merkletree::merkle::{
    get_merkle_tree_len_generic, Element, FromIndexedParallelIterator, MerkleTree,
};
use merkletree::store::{DiskStore, LevelCacheStore, StoreConfig, VecStore};
use std::path::PathBuf;
use typenum::{Unsigned, U0, U8};

/// Constructors
fn lc_instantiate_try_from_iter_with_config<
    E: Element,
    A: Algorithm<E>,
    BaseTreeArity: Unsigned,
>(
    leaves: usize,
    temp_dir_path: &PathBuf,
    rows_to_discard: usize,
) -> MerkleTree<E, A, LevelCacheStore<E, std::fs::File>, BaseTreeArity> {
    let dataset = generate_vector_of_elements::<E>(leaves);

    let replica_path = StoreConfig::data_path(temp_dir_path, "replica_path");
    let mut replica_file = std::fs::File::create(&replica_path)
        .expect("failed to create replica file [lc_instantiate_try_from_iter_with_config]");

    // prepare replica file content
    let config = StoreConfig::new(
        temp_dir_path,
        "lc_instantiate_try_from_iter_with_config",
        rows_to_discard,
    );

    // instantiation of this temp tree is required for binding config to actual file on disk for subsequent dumping the data to replica
    let tree = MerkleTree::<E, A, DiskStore<E>, BaseTreeArity>::try_from_iter_with_config(
        dataset.clone().into_iter().map(Ok),
        config.clone(),
    )
    .expect("failed to instantiate tree [lc_instantiate_try_from_iter_with_config]");

    dump_tree_data_to_replica::<E, BaseTreeArity>(
        tree.leafs(),
        tree.len(),
        &config,
        &mut replica_file,
    );

    // generate LC tree from dumped data
    let lc_config = StoreConfig::from_config(
        &config,
        format!("{}-{}", "lc_instantiate_try_from_iter_with_config", "lc"),
        Some(tree.len()),
    );
    let mut tree = MerkleTree::<E, A, LevelCacheStore<E, std::fs::File>, BaseTreeArity>::try_from_iter_with_config(dataset.into_iter().map(Ok), lc_config).expect("failed to instantiate LC tree [lc_instantiate_try_from_iter_with_config]");
    tree.set_external_reader_path(&replica_path)
        .expect("can't set external reader path [lc_instantiate_try_from_iter_with_config]");
    tree
}

fn lc_instantiate_from_par_iter_with_config<
    E: Element,
    A: Algorithm<E>,
    BaseTreeArity: Unsigned,
>(
    leaves: usize,
    temp_dir_path: &PathBuf,
    rows_to_discard: usize,
) -> MerkleTree<E, A, LevelCacheStore<E, std::fs::File>, BaseTreeArity> {
    let dataset = generate_vector_of_elements::<E>(leaves);

    let replica_path = StoreConfig::data_path(temp_dir_path, "replica_path");
    let mut replica_file = std::fs::File::create(&replica_path)
        .expect("failed to create replica file [lc_instantiate_from_par_iter_with_config]");

    // prepare replica file content
    let config = StoreConfig::new(
        temp_dir_path,
        "lc_instantiate_from_par_iter_with_config",
        rows_to_discard,
    );

    // instantiation of this temp tree is required for binding config to actual file on disk for subsequent dumping the data to replica
    let tree = MerkleTree::<E, A, DiskStore<E>, BaseTreeArity>::from_par_iter_with_config(
        dataset.clone(),
        config.clone(),
    )
    .expect("failed to instantiate tree [lc_instantiate_from_par_iter_with_config]");

    dump_tree_data_to_replica::<E, BaseTreeArity>(
        tree.leafs(),
        tree.len(),
        &config,
        &mut replica_file,
    );

    // generate LC tree from dumped data
    let lc_config = StoreConfig::from_config(
        &config,
        format!("{}-{}", "lc_instantiate_from_par_iter_with_config", "lc"),
        Some(tree.len()),
    );
    let mut tree = MerkleTree::<E, A, LevelCacheStore<E, std::fs::File>, BaseTreeArity>::from_par_iter_with_config(dataset, lc_config).expect("failed to instantiate LC tree [lc_instantiate_from_par_iter_with_config]");
    tree.set_external_reader_path(&replica_path)
        .expect("can't set external reader path [lc_instantiate_from_par_iter_with_config]");
    tree
}

fn lc_instantiate_from_data_with_config<E: Element, A: Algorithm<E>, BaseTreeArity: Unsigned>(
    leaves: usize,
    temp_dir_path: &PathBuf,
    rows_to_discard: usize,
) -> MerkleTree<E, A, LevelCacheStore<E, std::fs::File>, BaseTreeArity> {
    let dataset = generate_vector_of_usizes(leaves);

    let replica_path = StoreConfig::data_path(temp_dir_path, "replica_path");
    let mut replica_file = std::fs::File::create(&replica_path)
        .expect("failed to create replica file [lc_instantiate_from_data_with_config]");

    // prepare replica file content
    let config = StoreConfig::new(
        temp_dir_path,
        "lc_instantiate_from_data_with_config",
        rows_to_discard,
    );

    // instantiation of this temp tree is required for binding config to actual file on disk for subsequent dumping the data to replica
    let tree = MerkleTree::<E, A, DiskStore<E>, BaseTreeArity>::from_data_with_config(
        dataset.as_slice(),
        config.clone(),
    )
    .expect("failed to instantiate tree [lc_instantiate_from_data_with_config]");

    dump_tree_data_to_replica::<E, BaseTreeArity>(
        tree.leafs(),
        tree.len(),
        &config,
        &mut replica_file,
    );

    // generate LC tree from dumped data
    let lc_config = StoreConfig::from_config(
        &config,
        format!("{}-{}", "lc_instantiate_from_data_with_config", "lc"),
        Some(tree.len()),
    );
    let mut tree = MerkleTree::<E, A, LevelCacheStore<E, std::fs::File>, BaseTreeArity>::from_data_with_config(dataset.as_slice(), lc_config).expect("failed to instantiate LC tree [lc_instantiate_from_data_with_config]");
    tree.set_external_reader_path(&replica_path)
        .expect("can't set external reader path [lc_instantiate_from_data_with_config]");
    tree
}

fn lc_instantiate_from_byte_slice_with_config<
    E: Element,
    A: Algorithm<E>,
    BaseTreeArity: Unsigned,
>(
    leaves: usize,
    temp_dir_path: &PathBuf,
    rows_to_discard: usize,
) -> MerkleTree<E, A, LevelCacheStore<E, std::fs::File>, BaseTreeArity> {
    let dataset = generate_byte_slice_tree::<E, A>(leaves);

    let replica_path = StoreConfig::data_path(temp_dir_path, "replica_path");
    let mut replica_file = std::fs::File::create(&replica_path)
        .expect("failed to create replica file [lc_instantiate_from_byte_slice_with_config]");

    // prepare replica file content
    let config = StoreConfig::new(
        temp_dir_path,
        "lc_instantiate_from_byte_slice_with_config",
        rows_to_discard,
    );

    // instantiation of this temp tree is required for binding config to actual file on disk for subsequent dumping the data to replica
    let tree = MerkleTree::<E, A, DiskStore<E>, BaseTreeArity>::from_byte_slice_with_config(
        dataset.as_slice(),
        config.clone(),
    )
    .expect("failed to instantiate tree [lc_instantiate_from_byte_slice_with_config]");

    dump_tree_data_to_replica::<E, BaseTreeArity>(
        tree.leafs(),
        tree.len(),
        &config,
        &mut replica_file,
    );

    // generate LC tree from dumped data
    let lc_config = StoreConfig::from_config(
        &config,
        format!("{}-{}", "lc_instantiate_from_byte_slice_with_config", "lc"),
        Some(tree.len()),
    );
    let mut tree = MerkleTree::<E, A, LevelCacheStore<E, std::fs::File>, BaseTreeArity>::from_byte_slice_with_config(dataset.as_slice(), lc_config).expect("failed to instantiate LC tree [lc_instantiate_from_byte_slice_with_config]");
    tree.set_external_reader_path(&replica_path)
        .expect("can't set external reader path [lc_instantiate_from_byte_slice_with_config]");
    tree
}

fn lc_instantiate_from_tree_slice_with_config<
    E: Element,
    A: Algorithm<E>,
    BaseTreeArity: Unsigned,
>(
    leaves: usize,
    temp_dir_path: &PathBuf,
    rows_to_discard: usize,
) -> MerkleTree<E, A, LevelCacheStore<E, std::fs::File>, BaseTreeArity> {
    let dataset = generate_vector_of_usizes(leaves);
    let tmp_tree = MerkleTree::<E, A, VecStore<E>, BaseTreeArity>::from_data(dataset.as_slice())
        .expect("failed to instantiate tree [lc_instantiate_from_tree_slice_with_config]");
    let dataset = serialize_tree(tmp_tree);

    let replica_path = StoreConfig::data_path(temp_dir_path, "replica_path");
    let mut replica_file = std::fs::File::create(&replica_path)
        .expect("failed to create replica file [lc_instantiate_from_tree_slice_with_config]");

    // prepare replica file content
    let config = StoreConfig::new(
        temp_dir_path,
        "lc_instantiate_from_tree_slice_with_config",
        rows_to_discard,
    );

    // instantiation of this temp tree is required for binding config to actual file on disk for subsequent dumping the data to replica
    let tree = MerkleTree::<E, A, DiskStore<E>, BaseTreeArity>::from_tree_slice_with_config(
        dataset.as_slice(),
        leaves,
        config.clone(),
    )
    .expect("failed to instantiate tree [lc_instantiate_from_tree_slice_with_config]");

    dump_tree_data_to_replica::<E, BaseTreeArity>(
        tree.leafs(),
        tree.len(),
        &config,
        &mut replica_file,
    );

    // generate LC tree from dumped data
    let lc_config = StoreConfig::from_config(
        &config,
        format!("{}-{}", "lc_instantiate_from_tree_slice_with_config", "lc"),
        Some(tree.len()),
    );
    let mut tree = MerkleTree::<E, A, LevelCacheStore<E, std::fs::File>, BaseTreeArity>::from_tree_slice_with_config(dataset.as_slice(), leaves, lc_config).expect("failed to instantiate LC tree [lc_instantiate_from_tree_slice_with_config]");
    tree.set_external_reader_path(&replica_path)
        .expect("can't set external reader path [lc_instantiate_from_tree_slice_with_config]");
    tree
}

/// Test executor
fn run_test_base_lc_tree<E: Element, A: Algorithm<E>, BaseTreeArity: Unsigned>(
    constructor: fn(
        usize,
        &PathBuf,
        usize,
    ) -> MerkleTree<E, A, LevelCacheStore<E, std::fs::File>, BaseTreeArity>,
    leaves_in_tree: usize,
    temp_dir_path: &PathBuf,
    rows_to_discard: usize,
    expected_leaves: usize,
    expected_len: usize,
    expected_root: E,
) {
    // base tree has SubTreeArity and TopTreeArity parameters equal to zero
    let tree: MerkleTree<E, A, LevelCacheStore<E, std::fs::File>, BaseTreeArity> =
        constructor(leaves_in_tree, temp_dir_path, rows_to_discard);
    test_levelcache_tree_functionality(
        tree,
        Some(rows_to_discard),
        expected_leaves,
        expected_len,
        expected_root,
    );
}

/// LevelCache (base) trees can be instantiated only with constructors that take
/// valid configuration as input parameter, so we use only 'with_config' constructors
/// for testing. Again dataset is critical for root computation, so we use distinct
/// integration tests for different datasets
#[test]
fn test_base_levelcache_trees_iterable() {
    fn run_tests<E: Element + Copy, A: Algorithm<E>>(root: E) {
        let base_tree_leaves = 64;
        let expected_total_leaves = base_tree_leaves;
        type OctTree = U8;
        let len = get_merkle_tree_len_generic::<OctTree, U0, U0>(base_tree_leaves).unwrap();
        let rows_to_discard = 0;

        let distinguisher = "instantiate_ctree_from_store_configs_and_replica";
        let temp_dir = tempdir::TempDir::new(distinguisher).unwrap();
        run_test_base_lc_tree::<E, A, OctTree>(
            lc_instantiate_try_from_iter_with_config,
            base_tree_leaves,
            &temp_dir.as_ref().to_path_buf(),
            rows_to_discard,
            expected_total_leaves,
            len,
            root,
        );

        let distinguisher = "lc_instantiate_from_par_iter_with_config";
        let temp_dir = tempdir::TempDir::new(distinguisher).unwrap();
        run_test_base_lc_tree::<E, A, OctTree>(
            lc_instantiate_from_par_iter_with_config,
            base_tree_leaves,
            &temp_dir.as_ref().to_path_buf(),
            rows_to_discard,
            expected_total_leaves,
            len,
            root,
        );
    }

    let root_xor128 =
        TestItemType::from_slice(&[65, 0, 64, 0, 64, 0, 64, 0, 64, 0, 64, 0, 64, 0, 64, 0]);
    run_tests::<TestItemType, TestXOR128>(root_xor128);

    let root_sha256 = TestItemType::from_slice(&[
        252, 61, 163, 229, 140, 223, 198, 165, 200, 137, 59, 43, 83, 136, 197, 63,
    ]);
    run_tests::<TestItemType, TestSha256Hasher>(root_sha256);
}

#[test]
fn test_base_levelcache_trees_iterable_hashable_and_serialization() {
    fn run_tests<E: Element + Copy, A: Algorithm<E>>(root: E) {
        let base_tree_leaves = 64;
        let expected_total_leaves = base_tree_leaves;
        type OctTree = U8;
        let len = get_merkle_tree_len_generic::<OctTree, U0, U0>(base_tree_leaves).unwrap();
        let rows_to_discard = 0;

        let distinguisher = "lc_instantiate_from_data_with_config";
        let temp_dir = tempdir::TempDir::new(distinguisher).unwrap();
        run_test_base_lc_tree::<E, A, OctTree>(
            lc_instantiate_from_data_with_config,
            base_tree_leaves,
            &temp_dir.as_ref().to_path_buf(),
            rows_to_discard,
            expected_total_leaves,
            len,
            root,
        );

        let distinguisher = "lc_instantiate_from_byte_slice_with_config";
        let temp_dir = tempdir::TempDir::new(distinguisher).unwrap();
        run_test_base_lc_tree::<E, A, OctTree>(
            lc_instantiate_from_byte_slice_with_config,
            base_tree_leaves,
            &temp_dir.as_ref().to_path_buf(),
            rows_to_discard,
            expected_total_leaves,
            len,
            root,
        );

        /* TODO investigate, why this test fails
        let distinguisher = "lc_instantiate_from_tree_slice_with_config";
        let temp_dir = tempdir::TempDir::new(distinguisher).unwrap();
        run_test_base_lc_tree::<E, A, OctTree>(
            lc_instantiate_from_tree_slice_with_config,
            base_tree_leaves,
            &temp_dir.as_ref().to_path_buf(),
            rows_to_discard,
            expected_total_leaves,
            len,
            root,
        );
        */
    }

    let root_xor128 = TestItemType::from_slice(&[1, 0, 0, 0, 19, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    run_tests::<TestItemType, TestXOR128>(root_xor128);

    let root_sha256 = TestItemType::from_slice(&[
        98, 103, 202, 101, 121, 179, 6, 237, 133, 39, 253, 169, 173, 63, 89, 188,
    ]);
    run_tests::<TestItemType, TestSha256Hasher>(root_sha256);
}
