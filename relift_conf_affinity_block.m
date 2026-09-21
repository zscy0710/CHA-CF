function A_conf = relift_conf_affinity_block(block, state, dev_data, fit_data)
%RELIFT_CONF_AFFINITY_BLOCK 计算某个局部子树块的后验相似度矩阵。
% 相对于 root-only 版本，这里只看 block.parent 下面那棵子树内部的混淆。

labels = block.pair_leaf_labels(:)';
num_labels = numel(labels);
A_conf = zeros(num_labels, num_labels);

sample_idx = ismember(dev_data(:, end), relift_leaf_descendants(state.tree, block.parent));
if ~any(sample_idx)
    return;
end

oracle_pred = relift_predict_from_node(dev_data(sample_idx,1:end-1), state, block.parent);
local_true = dev_data(sample_idx, end);
class_counts = accumarray(fit_data(:, end), 1, [max(fit_data(:, end)) 1]);
weights = class_counts(local_true) .^ (-1.0);

for i = 1:numel(local_true)
    ti = find(labels == local_true(i), 1);
    pj = find(labels == oracle_pred(i), 1);
    if ~isempty(ti) && ~isempty(pj) && ti ~= pj
        A_conf(ti, pj) = A_conf(ti, pj) + weights(i);
    end
end

for i = 1:num_labels
    row_sum = sum(A_conf(i, :));
    if row_sum > 0
        A_conf(i, :) = A_conf(i, :) / row_sum;
    end
end
A_conf = 0.5 * (A_conf + A_conf');
A_conf(1:size(A_conf,1)+1:end) = 0;
end
