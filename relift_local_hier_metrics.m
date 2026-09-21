function metrics = relift_local_hier_metrics(tree, node_id, true_label, pred_label)
%RELIFT_LOCAL_HIER_METRICS Compute local F_H and normalized TIE.

leaf_ids = relift_leaf_descendants(tree, node_id);
sample_idx = ismember(true_label, leaf_ids);
metrics = struct('FH', 0, 'TIE_norm', 1);
if ~any(sample_idx)
    return;
end

local_true = true_label(sample_idx);
local_pred = pred_label(sample_idx);
count = numel(local_true);
max_distance = max(relift_subtree_max_distance(tree, node_id), 1);
sum_fh = 0;
sum_tie = 0;

for i = 1:count
    path_true = relift_local_path(tree, node_id, local_true(i));
    path_pred = relift_local_path(tree, node_id, local_pred(i));
    overlap = intersect(path_true, path_pred);

    precision = numel(overlap) / max(numel(path_pred), 1);
    recall = numel(overlap) / max(numel(path_true), 1);
    if precision + recall > 0
        sum_fh = sum_fh + 2 * precision * recall / (precision + recall);
    end
    sum_tie = sum_tie + numel(setxor(path_true, path_pred));
end

metrics.FH = sum_fh / count;
metrics.TIE_norm = (sum_tie / count) / max_distance;
end

function path_nodes = relift_local_path(tree, node_id, leaf_id)
path_nodes = leaf_id;
current = leaf_id;
while current ~= node_id && current ~= 0
    current = tree(current, 1);
    if current ~= 0
        path_nodes(end+1) = current; %#ok<AGROW>
    end
end
end

function max_distance = relift_subtree_max_distance(tree, node_id)
leaf_ids = relift_leaf_descendants(tree, node_id);
if numel(leaf_ids) <= 1
    max_distance = 1;
    return;
end

max_distance = 1;
for i = 1:numel(leaf_ids)
    for j = i+1:numel(leaf_ids)
        path_i = relift_local_path(tree, node_id, leaf_ids(i));
        path_j = relift_local_path(tree, node_id, leaf_ids(j));
        max_distance = max(max_distance, numel(setxor(path_i, path_j)));
    end
end
end
