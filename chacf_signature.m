function key = chacf_signature(tree, node)
%CHACF_SIGNATURE Make a stable key from the leaf labels below a node.

leaves = sort(unique(chacf_descendants(tree, node)));
key = sprintf('%d_', leaves);
key(end) = [];
end
