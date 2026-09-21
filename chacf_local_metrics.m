function [fh, tie] = chacf_local_metrics(tree, node, truth, pred)
%CHACF_LOCAL_METRICS Compute local F_H and normalized TIE.

leaves = chacf_descendants(tree, node);
index = ismember(truth, leaves);
if ~any(index)
    fh = 0;
    tie = 1;
    return;
end

truth = truth(index);
pred = pred(index);
maxDistance = local_max_distance(tree, node);
sumFH = 0;
sumTIE = 0;

for i = 1:numel(truth)
    truePath = local_path(tree, node, truth(i));
    predPath = local_path(tree, node, pred(i));
    common = intersect(truePath, predPath);
    p = numel(common) / numel(predPath);
    r = numel(common) / numel(truePath);
    if p + r > 0
        sumFH = sumFH + 2 * p * r / (p + r);
    end
    sumTIE = sumTIE + numel(setxor(truePath, predPath));
end

fh = sumFH / numel(truth);
tie = sumTIE / numel(truth) / maxDistance;
end

function path = local_path(tree, node, leaf)
path = leaf;
while path(end) ~= node && path(end) ~= 0
    parent = tree(path(end), 1);
    if parent ~= 0
        path(end+1) = parent;
    else
        path(end+1) = 0;
    end
end
path(path == 0) = [];
end

function d = local_max_distance(tree, node)
leaves = chacf_descendants(tree, node);
d = 1;
for i = 1:numel(leaves)
    for j = i+1:numel(leaves)
        d = max(d, numel(setxor( ...
            local_path(tree, node, leaves(i)), local_path(tree, node, leaves(j)))));
    end
end
end
