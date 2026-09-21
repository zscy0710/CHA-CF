function child = chacf_child_on_path(tree, parent, leaf)
%CHACF_CHILD_ON_PATH Return the direct child below a parent.

child = leaf;
while tree(child, 1) ~= parent && tree(child, 1) ~= 0
    child = tree(child, 1);
end
end
