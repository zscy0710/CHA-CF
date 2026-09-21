function affinity = chacf_confusion_affinity(candidate, state, devData, fitData)
%CHACF_CONFUSION_AFFINITY Compute confusion-based affinity for a candidate block.

labels = candidate.leaves(:)';
affinity = zeros(numel(labels));
index = ismember(devData(:, end), chacf_descendants(state.tree, candidate.parent));
if ~any(index)
    return;
end

pred = chacf_predict(devData(index, 1:end-1), state, candidate.parent);
truth = devData(index, end);
classCount = accumarray(fitData(:, end), 1, [max(fitData(:, end)), 1]);
weight = classCount(truth) .^ -1;

for i = 1:numel(truth)
    r = find(labels == truth(i), 1);
    c = find(labels == pred(i), 1);
    if ~isempty(r) && ~isempty(c) && r ~= c
        affinity(r, c) = affinity(r, c) + weight(i);
    end
end

for i = 1:numel(labels)
    rowSum = sum(affinity(i, :));
    if rowSum > 0
        affinity(i, :) = affinity(i, :) / rowSum;
    end
end
affinity = (affinity + affinity') / 2;
affinity(1:size(affinity, 1)+1:end) = 0;
end
