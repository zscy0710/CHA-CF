function child_id = relift_child_on_path(tree, parent_node, leaf_id)
%RELIFT_CHILD_ON_PATH 找到某个叶子路径上 parent_node 对应的直接孩子。

child_id = 0;
current = leaf_id;

while current ~= 0
    parent = tree(current, 1);
    if parent == parent_node
        child_id = current;
        return;
    end
    current = parent;
end
end
