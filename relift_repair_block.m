function repaired_tree = relift_repair_block(tree, compact_train, block, state, dev_data, fit_data, fuse_lambda)
%RELIFT_REPAIR_BLOCK 重构任意内部节点下选中的局部块。
% 逻辑和 root-only 版本一致，只是替换位置从 root 扩到 block.parent。
% 当前 block 既可以是一对孩子的 pair candidate，也可以是整父节点重分组的 whole-parent candidate。

if nargin < 7
    fuse_lambda = 0.7;
end

A_orig = relift_orig_affinity(compact_train, block.pair_leaf_labels);
A_conf = relift_conf_affinity_block(block, state, dev_data, fit_data);
A_star = fuse_lambda * A_orig + (1 - fuse_lambda) * A_conf;
new_groups = relift_cluster_affinity(A_star, block.pair_leaf_labels, numel(block.pair_children));

root = tree_Root(tree);
root_spec = relift_tree_to_spec(tree, root);
root_spec = relift_replace_block_in_spec(root_spec, block.parent, block.pair_children, new_groups);
repaired_tree = relift_spec_to_tree(root_spec);
end
