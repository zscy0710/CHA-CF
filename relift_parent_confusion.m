function [raw_conf, children, total_weight] = relift_parent_confusion(tree, parent_node, state, dev_data, fit_data)
%RELIFT_PARENT_CONFUSION Compute weighted child-level confusion under one parent.

children = get_children_set(tree, parent_node);
leaf_ids = relift_leaf_descendants(tree, parent_node);
sample_idx = ismember(dev_data(:, end), leaf_ids);

raw_conf = zeros(numel(children), numel(children));
total_weight = 0;
if numel(children) < 2 || ~any(sample_idx)
    return;
end

oracle_pred = relift_predict_from_node(dev_data(sample_idx, 1:end-1), state, parent_node);
local_true = dev_data(sample_idx, end);
class_counts = accumarray(fit_data(:, end), 1, [max(fit_data(:, end)) 1]);
weights = class_counts(local_true) .^ (-1.0);
total_weight = sum(weights);

for i = 1:numel(local_true)
    true_child = relift_child_on_path(tree, parent_node, local_true(i));
    pred_child = relift_child_on_path(tree, parent_node, oracle_pred(i));
    true_idx = find(children == true_child, 1);
    pred_idx = find(children == pred_child, 1);
    if ~isempty(true_idx) && ~isempty(pred_idx) && true_idx ~= pred_idx
        raw_conf(true_idx, pred_idx) = raw_conf(true_idx, pred_idx) + weights(i);
    end
end
end
