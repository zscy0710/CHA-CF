function [sym_conf, pair_children, pair_leaf_labels, raw_conf, children, total_weight] = relift_parent_confusion(tree, parent_node, state, dev_data, fit_data)
%RELIFT_PARENT_CONFUSION 统计某个父节点下孩子之间的混淆。
% 这个函数只在 parent_node 对应的子树内部工作。
% 返回内容分成两层：
% 1. sym_conf：按行归一化后再对称化的局部混淆矩阵，用来描述结构关系。
% 2. raw_conf：原始加权混淆矩阵，用来计算 pair 和 whole-parent 的绝对错误质量。
% 同时返回该父节点下面最混淆的一对孩子，以及这对孩子对应的叶标签集合。

children = get_children_set(tree, parent_node);
leaf_ids = relift_leaf_descendants(tree, parent_node);
sample_idx = ismember(dev_data(:, end), leaf_ids);

sym_conf = zeros(numel(children), numel(children));
raw_conf = zeros(numel(children), numel(children));
pair_children = [];
pair_leaf_labels = [];
total_weight = 0;
if numel(children) < 2 || ~any(sample_idx)
    return;
end

oracle_pred = relift_predict_from_node(dev_data(sample_idx, 1:end-1), state, parent_node);
local_true = dev_data(sample_idx, end);
class_counts = accumarray(fit_data(:, end), 1, [max(fit_data(:, end)) 1]);
weights = class_counts(local_true) .^ (-1.0);
total_weight = sum(weights);

% 先保留原始加权混淆矩阵，后面 pair candidate 和 whole-parent candidate
% 都用这份矩阵来计算绝对错误质量，避免量纲不一致。
for i = 1:numel(local_true)
    true_child = relift_child_on_path(tree, parent_node, local_true(i));
    pred_child = relift_child_on_path(tree, parent_node, oracle_pred(i));
    ti = find(children == true_child, 1);
    pj = find(children == pred_child, 1);
    if ~isempty(ti) && ~isempty(pj) && ti ~= pj
        raw_conf(ti, pj) = raw_conf(ti, pj) + weights(i);
    end
end

sym_conf = raw_conf;
row_sum = sum(sym_conf, 2);
for i = 1:size(sym_conf, 1)
    if row_sum(i) > 0
        sym_conf(i, :) = sym_conf(i, :) / row_sum(i);
    end
end
sym_conf = 0.5 * (sym_conf + sym_conf');
sym_conf(1:size(sym_conf,1)+1:end) = 0;

% pair 候选不再只看归一化后的最大值，而是看它实际吃掉了多少加权错误。
pair_mass = raw_conf + raw_conf';
pair_mass(1:size(pair_mass,1)+1:end) = 0;
[~, idx] = max(pair_mass(:));
[ri, ci] = ind2sub(size(pair_mass), idx);
if isempty(ri) || ri == ci || pair_mass(ri, ci) <= 0
    return;
end

pair_children = children([ri, ci]);
for i = 1:numel(pair_children)
    pair_leaf_labels = [pair_leaf_labels; relift_leaf_descendants(tree, pair_children(i))]; %#ok<AGROW>
end
pair_leaf_labels = unique(pair_leaf_labels);
end
