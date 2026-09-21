function [confusion, children, totalWeight] = chacf_parent_confusion(tree, parent, state, devData, fitData)
%CHACF_PARENT_CONFUSION Compute child-level confusion under one parent.

children = get_children_set(tree, parent);
leaves = chacf_descendants(tree, parent);
index = ismember(devData(:, end), leaves);
confusion = zeros(numel(children));
totalWeight = 0;
if numel(children) < 2 || ~any(index)
    return;
end

pred = chacf_predict(devData(index, 1:end-1), state, parent);
truth = devData(index, end);
classCount = accumarray(fitData(:, end), 1, [max(fitData(:, end)), 1]);
weight = classCount(truth) .^ -1;
totalWeight = sum(weight);

for i = 1:numel(truth)
    trueChild = chacf_child_on_path(tree, parent, truth(i));
    predChild = chacf_child_on_path(tree, parent, pred(i));
    r = find(children == trueChild, 1);
    c = find(children == predChild, 1);
    if ~isempty(r) && ~isempty(c) && r ~= c
        confusion(r, c) = confusion(r, c) + weight(i);
    end
end
end
