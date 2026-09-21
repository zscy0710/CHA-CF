function child = relift_child_on_path(tree, parent_node, leaf_id)
%RELIFT_CHILD_ON_PATH Return the direct child of PARENT_NODE on a leaf path.

child = leaf_id;
while tree(child, 1) ~= parent_node && tree(child, 1) ~= 0
    child = tree(child, 1);
end
end
