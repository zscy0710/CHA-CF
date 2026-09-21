function signature = relift_node_signature(tree, node_id)
%RELIFT_NODE_SIGNATURE Identify a subtree by its leaf labels.

leaf_ids = sort(unique(relift_leaf_descendants(tree, node_id)));
parts = strings(1, numel(leaf_ids));
for i = 1:numel(leaf_ids)
    parts(i) = string(leaf_ids(i));
end
signature = char(strjoin(parts, '_'));
end
