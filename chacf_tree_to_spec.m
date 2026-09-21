function spec = chacf_tree_to_spec(tree, node)
%CHACF_TREE_TO_SPEC Convert a matrix tree to a nested representation.

if ismember(node, tree_LeafNode(tree))
    spec = node;
    return;
end

children = get_children_set(tree, node);
subtree = cell(1, numel(children));
for i = 1:numel(children)
    subtree{i} = chacf_tree_to_spec(tree, children(i));
end
spec = struct('id', node, 'children', {subtree});
end
