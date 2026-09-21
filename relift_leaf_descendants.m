function leaf_ids = relift_leaf_descendants(tree, node_id)
%RELIFT_LEAF_DESCENDANTS Return all leaf labels below a node.

leaf_ids = [];
children = get_children_set(tree, node_id);
if isempty(children)
    leaf_ids = node_id;
    return;
end

for i = 1:numel(children)
    leaf_ids = [leaf_ids; relift_leaf_descendants(tree, children(i))]; %#ok<AGROW>
end
end
