function [best_block, best_tree, best_dev_score, decision] = relift_select_block_by_dev(current_tree, train_data, blocks, fit_state, fit_data, dev_data, dataset_name, fuse_lambda, cfg)
%RELIFT_SELECT_BLOCK_BY_DEV Select a candidate using the development split.

eval_topk = min(cfg.parentpair_dev_eval_topk, numel(blocks));
base_pred = local_predict_dev_labels(dev_data(:, 1:end-1), fit_state, current_tree);
base_hier_metrics = relift_local_hier_metrics( ...
    current_tree, tree_Root(current_tree), dev_data(:, end), base_pred);
base_dev_score = mean(dev_data(:, end) == base_pred) + ...
    base_hier_metrics.FH - base_hier_metrics.TIE_norm;

best_dev_score = -Inf;
best_block = blocks(1);
best_tree = current_tree;
for candidate_id = 1:eval_topk
    block = blocks(candidate_id);
    temp_tree = relift_repair_block( ...
        current_tree, train_data, block, fit_state, dev_data, fit_data, fuse_lambda);
    temp_state = relift_train_state(fit_data, temp_tree, dataset_name, cfg);
    temp_pred = local_predict_dev_labels(dev_data(:, 1:end-1), temp_state, temp_tree);

    hier_metrics = relift_local_hier_metrics( ...
        temp_tree, tree_Root(temp_tree), dev_data(:, end), temp_pred);
    dev_score = mean(dev_data(:, end) == temp_pred) + ...
        hier_metrics.FH - hier_metrics.TIE_norm;

    if dev_score > best_dev_score
        best_dev_score = dev_score;
        best_block = block;
        best_tree = temp_tree;
    end
end

decision = struct();
decision.base_dev_score = base_dev_score;
decision.best_dev_score = best_dev_score;
decision.delta_dev_score = best_dev_score - base_dev_score;
decision.accepted = best_dev_score > base_dev_score + cfg.accept_min_delta;
if ~decision.accepted
    best_tree = current_tree;
end
end

function pred = local_predict_dev_labels(eval_x, state, tree)
pred = FS_topDownSVMPrediction( ...
    eval_x, state.models, tree, state.feature_slct, state.numberSel, 0)';
pred = double(pred(:));
end
