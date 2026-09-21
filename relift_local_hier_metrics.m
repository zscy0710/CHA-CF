function metrics = relift_local_hier_metrics(tree, node_id, true_label, pred_label)
%RELIFT_LOCAL_HIER_METRICS 计算某个子树内部的局部层次指标。
% 这里主要给 holdout 上的候选块复核提供 FH 和 TIE 等层次分数。

leaf_ids = relift_leaf_descendants(tree, node_id);
sample_idx = ismember(true_label, leaf_ids);

metrics = struct('FH', 0, 'TIE', 1, 'TIE_norm', 1, 'count', 0);
if ~any(sample_idx)
    return;
end

local_true = true_label(sample_idx);
local_pred = pred_label(sample_idx);
count = numel(local_true);

sumPH = 0;
sumRH = 0;
sumFH = 0;
sumTIE = 0;
maxD = relift_subtree_max_distance(tree, node_id);
maxD = max(maxD, 1);

% 对每个样本，比较它在该子树内的真实路径和预测路径。
for i = 1:count
    path_true = relift_local_path(tree, node_id, local_true(i));
    path_pred = relift_local_path(tree, node_id, local_pred(i));
    inter = intersect(path_true, path_pred);

    PH = numel(inter) / max(numel(path_pred), 1);
    RH = numel(inter) / max(numel(path_true), 1);
    if PH + RH > 0
        FH = 2 * PH * RH / (PH + RH);
    else
        FH = 0;
    end

    tie_i = numel(setxor(path_true, path_pred));
    sumPH = sumPH + PH;
    sumRH = sumRH + RH;
    sumFH = sumFH + FH;
    sumTIE = sumTIE + tie_i;
end

metrics.PH = sumPH / count;
metrics.RH = sumRH / count;
metrics.FH = sumFH / count;
metrics.TIE = sumTIE / count;
metrics.TIE_norm = metrics.TIE / maxD;
metrics.count = count;

% 同时补一份该子树内部的宏观混淆统计。
labels = unique([local_true(:); local_pred(:)]);
conf = zeros(numel(labels));
for i = 1:count
    r = find(labels == local_true(i), 1);
    c = find(labels == local_pred(i), 1);
    conf(r, c) = conf(r, c) + 1;
end

tp = diag(conf);
fp = sum(conf, 1)' - tp;
fn = sum(conf, 2) - tp;
precision = tp ./ max(tp + fp, 1);
recall = tp ./ max(tp + fn, 1);
f1 = 2 * precision .* recall ./ max(precision + recall, eps);

metrics.RH_balanced = mean(recall);
metrics.FH_macro = mean(f1);

train_counts = zeros(numel(labels), 1);
for i = 1:numel(labels)
    train_counts(i) = sum(local_true == labels(i));
end

tail_idx = train_counts <= max(train_counts) * 0.2;
if ~any(tail_idx)
    tail_idx = train_counts == min(train_counts);
end
metrics.tail_recall = mean(recall(tail_idx));

off_diag = conf;
off_diag(1:numel(labels)+1:end) = 0;
off_mass = sum(off_diag(:));
if off_mass > 0
    p = off_diag(off_diag > 0) / off_mass;
    metrics.confusion_entropy = -sum(p .* log(p)) / log(max(numel(p), 2));
else
    metrics.confusion_entropy = 0;
end
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

function maxD = relift_subtree_max_distance(tree, node_id)

leaf_ids = relift_leaf_descendants(tree, node_id);
if numel(leaf_ids) <= 1
    maxD = 1;
    return;
end

maxD = 1;
for i = 1:numel(leaf_ids)
    for j = i+1:numel(leaf_ids)
        path_i = relift_local_path(tree, node_id, leaf_ids(i));
        path_j = relift_local_path(tree, node_id, leaf_ids(j));
        d = numel(setxor(path_i, path_j));
        if d > maxD
            maxD = d;
        end
    end
end
end
