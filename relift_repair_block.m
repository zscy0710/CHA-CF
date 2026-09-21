function repaired_tree = relift_repair_block(tree, train_data, block, state, dev_data, fit_data, fuse_lambda)
%RELIFT_REPAIR_BLOCK Rebuild the selected child pair under its parent node.

A_orig = relift_orig_affinity(train_data, block.pair_leaf_labels);
A_conf = relift_conf_affinity_block(block, state, dev_data, fit_data);
A_star = fuse_lambda * A_orig + (1 - fuse_lambda) * A_conf;
new_groups = relift_cluster_affinity( ...
    A_star, block.pair_leaf_labels, numel(block.pair_children));

root = tree_Root(tree);
root_spec = relift_tree_to_spec(tree, root);
root_spec = relift_replace_block_in_spec( ...
    root_spec, block.parent, block.pair_children, new_groups);
repaired_tree = relift_spec_to_tree(root_spec);
end
