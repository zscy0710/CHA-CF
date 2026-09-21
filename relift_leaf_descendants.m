function leaf_ids = relift_leaf_descendants(tree, node_id)
%RELIFT_LEAF_DESCENDANTS 取出某个节点下面的全部叶子标签。

leaf_nodes = tree_LeafNode(tree);
if ismember(node_id, leaf_nodes)
    leaf_ids = node_id;
    return;
end

descendants = tree_Descendant(tree, node_id);
leaf_ids = descendants(ismember(descendants, leaf_nodes));
end
