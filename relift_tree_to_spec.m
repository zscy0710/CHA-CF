function spec = relift_tree_to_spec(tree, node_id)
%RELIFT_TREE_TO_SPEC Convert the matrix representation to a nested tree.

leaf_nodes = tree_LeafNode(tree);
if ismember(node_id, leaf_nodes)
    spec = node_id;
    return;
end

children = get_children_set(tree, node_id);
child_specs = cell(1, numel(children));
for i = 1:numel(children)
    child_specs{i} = relift_tree_to_spec(tree, children(i));
end

spec = struct('id', node_id, 'children', {child_specs});
end
