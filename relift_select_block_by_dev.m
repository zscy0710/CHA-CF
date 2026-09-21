function [best_block, best_tree, best_dev_score, decision] = relift_select_block_by_dev(current_tree, compact_train, blocks, fit_state, fit_data, dev_data, dataset_name, fuse_lambda, cfg)
%RELIFT_SELECT_BLOCK_BY_DEV Re-rank repair candidates on held-out data.
% Optional warm starts reuse MIMR solutions only for structurally identical
% node subproblems. They are disabled by default.

if ~isfield(cfg, 'accept_min_delta')
    cfg.accept_min_delta = 0;
end
use_warm = isfield(cfg, 'mimr_warm_start') && ~isempty(cfg.mimr_warm_start) && cfg.mimr_warm_start;
warm_cache = containers.Map('KeyType', 'char', 'ValueType', 'any');

eval_topk = min(cfg.parentpair_dev_eval_topk, numel(blocks));
base_pred = local_predict_dev_labels(dev_data(:, 1:end-1), fit_state, current_tree);
train_counts = accumarray(fit_data(:, end), 1, [max(fit_data(:, end)) 1]);
base_leaf_metrics = relift_eval_leaf_metrics(dev_data(:, end), base_pred, train_counts);
base_hier_metrics = relift_local_hier_metrics( ...
    current_tree, tree_Root(current_tree), dev_data(:, end), base_pred);
base_dev_score = base_leaf_metrics.acc + base_hier_metrics.FH - base_hier_metrics.TIE_norm;

best_dev_score = -Inf;
best_block = blocks(1);
best_tree = current_tree;
for candidate_id = 1:eval_topk
    block = blocks(candidate_id);
    temp_tree = relift_repair_block( ...
        current_tree, compact_train, block, fit_state, dev_data, fit_data, fuse_lambda);
    if use_warm
        [temp_state, W_full] = relift_train_state( ...
            fit_data, temp_tree, dataset_name, cfg, ...
            local_collect_warm(temp_tree, warm_cache));
        local_store_warm(temp_tree, W_full, warm_cache);
    else
        temp_state = relift_train_state(fit_data, temp_tree, dataset_name, cfg);
    end
    temp_pred = local_predict_dev_labels(dev_data(:, 1:end-1), temp_state, temp_tree);

    leaf_metrics = relift_eval_leaf_metrics(dev_data(:, end), temp_pred, train_counts);
    hier_metrics = relift_local_hier_metrics( ...
        temp_tree, tree_Root(temp_tree), dev_data(:, end), temp_pred);
    dev_score = leaf_metrics.acc + hier_metrics.FH - hier_metrics.TIE_norm;

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

function warm_W = local_collect_warm(tree, warm_cache)
%LOCAL_COLLECT_WARM Recover compatible MIMR initializations from the cache.
nodes = local_noleaf_nodes(tree);
warm_W = cell(max(nodes), 1);
for i = 1:numel(nodes)
    sig = relift_node_partition_signature(tree, nodes(i));
    if isKey(warm_cache, sig)
        warm_W{nodes(i)} = warm_cache(sig);
    end
end
end

function local_store_warm(tree, W_full, warm_cache)
%LOCAL_STORE_WARM Store a candidate's full MIMR solutions by subproblem key.
if isempty(W_full)
    return;
end
nodes = local_noleaf_nodes(tree);
for i = 1:numel(nodes)
    if numel(W_full) >= nodes(i) && ~isempty(W_full{nodes(i)})
        warm_cache(relift_node_partition_signature(tree, nodes(i))) = W_full{nodes(i)}; %#ok<NASGU>
    end
end
end

function nodes = local_noleaf_nodes(tree)
internal_nodes = tree_InternalNodes(tree);
internal_nodes(internal_nodes == -1) = [];
nodes = [internal_nodes; tree_Root(tree)];
end
