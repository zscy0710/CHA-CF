function metrics = relift_eval_leaf_metrics(true_label, pred_label, train_counts)
%RELIFT_EVAL_LEAF_METRICS 计算一个数据划分上的叶子层快速指标。
% 这些指标主要用于候选块在 holdout 上的轻量比较。

labels = unique([true_label(:); pred_label(:)]);
conf = zeros(max(labels), max(labels));
for i = 1:numel(true_label)
    conf(true_label(i), pred_label(i)) = conf(true_label(i), pred_label(i)) + 1;
end

support = sum(conf, 2);
pred_support = sum(conf, 1)';
tp = diag(conf);

precision = zeros(size(tp));
recall = zeros(size(tp));
f1 = zeros(size(tp));

precision(pred_support > 0) = tp(pred_support > 0) ./ pred_support(pred_support > 0);
recall(support > 0) = tp(support > 0) ./ support(support > 0);
valid = precision + recall > 0;
f1(valid) = 2 * precision(valid) .* recall(valid) ./ (precision(valid) + recall(valid));

valid_cls = support > 0;
acc = mean(true_label(:) == pred_label(:));
macro_f1 = mean(f1(valid_cls));
bal_acc = mean(recall(valid_cls));

% tail recall 只看训练集中最小的那部分标签。
train_counts = train_counts(:);
present_labels = find(train_counts > 0);
if isempty(present_labels)
    tail_recall = 0;
else
    [~, order] = sort(train_counts(present_labels), 'ascend');
    tail_num = max(1, round(numel(present_labels) * 0.3));
    tail_labels = present_labels(order(1:tail_num));
    tail_labels = tail_labels(support(tail_labels) > 0);
    if isempty(tail_labels)
        tail_recall = 0;
    else
        tail_recall = mean(recall(tail_labels));
    end
end

metrics = struct();
metrics.acc = acc;
metrics.macro_f1 = macro_f1;
metrics.bal_acc = bal_acc;
metrics.tail_recall = tail_recall;
end
