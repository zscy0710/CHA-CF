function A_orig = relift_orig_affinity(compact_train, leaf_labels)
%RELIFT_ORIG_AFFINITY 计算候选块内部的先验相似度矩阵。
% 这个矩阵来自 compact sample 本身的几何结构，是修树时的先验项。

labels = leaf_labels(:)';
num_labels = numel(labels);
A_orig = zeros(num_labels, num_labels);
counts = zeros(num_labels, 1);
proto = zeros(num_labels, size(compact_train, 2) - 1);

for i = 1:num_labels
    rows = compact_train(compact_train(:, end) == labels(i), 1:end-1);
    counts(i) = size(rows, 1);
    proto(i, :) = mean(rows, 1);
end

% 两两相似度由距离、相关性和简单的偏态项共同决定。
for i = 1:num_labels
    A_orig(i, i) = 1;
    for j = i+1:num_labels
        dist_term = 1 / (1 + norm(proto(i, :) - proto(j, :), 2));
        corr_term = corr(proto(i, :)', proto(j, :)', 'type', 'pearson');
        if isnan(corr_term)
            corr_term = 0;
        end
        corr_term = max(corr_term, 0);
        skew_term = 1 - abs(counts(i) - counts(j)) / (1 + abs(counts(i) - counts(j)));
        sample_term = counts(i) + counts(j);
        A_orig(i, j) = corr_term * dist_term * skew_term / max(sample_term, eps);
        A_orig(j, i) = A_orig(i, j);
    end
end
end
