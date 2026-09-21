function leaves = chacf_descendants(tree, node)
%CHACF_DESCENDANTS Return leaf labels below a node.

children = get_children_set(tree, node);
if isempty(children)
    leaves = node;
    return;
end

leaves = [];
for i = 1:numel(children)
    leaves = [leaves; chacf_descendants(tree, children(i))];
end
end
