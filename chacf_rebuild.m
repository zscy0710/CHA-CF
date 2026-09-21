function newTree = chacf_rebuild(tree, trainData, candidate, state, devData, fitData, fusion)
%CHACF_REBUILD Reconstruct one confused child pair.

A0 = chacf_feature_affinity(trainData, candidate.leaves);
A1 = chacf_confusion_affinity(candidate, state, devData, fitData);
A = fusion * A0 + (1 - fusion) * A1;
groups = chacf_cluster(A, candidate.leaves, numel(candidate.children));

spec = chacf_tree_to_spec(tree, tree_Root(tree));
spec = chacf_replace_block( ...
    spec, candidate.parent, candidate.children, groups);
newTree = chacf_spec_to_tree(spec);
end
