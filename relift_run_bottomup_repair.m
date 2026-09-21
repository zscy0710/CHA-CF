function [H1, applied_blocks, layer_progress] = relift_run_bottomup_repair(H0, train_data, fit_data, dev_data, dataset_name, fuse_lambda, cfg)
%RELIFT_RUN_BOTTOMUP_REPAIR Apply bottom-up local hierarchy adaptation.

current_tree = H0;
applied_blocks = struct('global_iter', {}, 'pass_idx', {}, 'layer_depth', {}, 'layer_round', {}, ...
    'parent', {}, 'parent_signature', {}, 'parent_leaf_labels', {}, ...
    'pair_children', {}, 'pair_leaf_labels', {}, 'score', {}, ...
    'dev_score_before', {}, 'dev_score_after', {}, 'delta_dev_score', {});
layer_progress = struct('pass_idx', {}, 'layer_depth', {}, 'num_parents_scanned', {}, ...
    'num_candidates', {}, 'num_candidates_evaluated', {}, 'num_blocks_applied', {}, ...
    'stop_reason', {});

global_iter = 0;
for pass_idx = 1:cfg.max_global_passes
    pass_applied = false;
    layer_map = relift_build_layer_map(current_tree);
    for layer_idx = 1:numel(layer_map)
        active_depth = layer_map(layer_idx).depth;
        layer_round = 0;
        layer_applied = 0;
        stop_reason = 'no_candidate';
        blocks = [];
        while layer_applied < cfg.max_blocks_per_layer
            fit_state = relift_train_state(fit_data, current_tree, dataset_name, cfg);
            blocks = relift_collect_parentpair_candidates( ...
                current_tree, fit_state, dev_data, fit_data, cfg, applied_blocks, active_depth);
            if isempty(blocks)
                break;
            end

            [best_block, best_tree, ~, decision] = relift_select_block_by_dev( ...
                current_tree, train_data, blocks, fit_state, fit_data, dev_data, ...
                dataset_name, fuse_lambda, cfg);
            if ~decision.accepted
                stop_reason = 'no_positive_gain';
                break;
            end

            current_tree = best_tree;
            layer_round = layer_round + 1;
            layer_applied = layer_applied + 1;
            pass_applied = true;
            global_iter = global_iter + 1;

            applied_blocks(end+1).global_iter = global_iter; %#ok<AGROW>
            applied_blocks(end).pass_idx = pass_idx;
            applied_blocks(end).layer_depth = active_depth;
            applied_blocks(end).layer_round = layer_round;
            applied_blocks(end).parent = best_block.parent;
            applied_blocks(end).parent_signature = best_block.parent_signature;
            applied_blocks(end).parent_leaf_labels = best_block.parent_leaf_labels;
            applied_blocks(end).pair_children = best_block.pair_children;
            applied_blocks(end).pair_leaf_labels = best_block.pair_leaf_labels;
            applied_blocks(end).score = best_block.score;
            applied_blocks(end).dev_score_before = decision.base_dev_score;
            applied_blocks(end).dev_score_after = decision.best_dev_score;
            applied_blocks(end).delta_dev_score = decision.delta_dev_score;

            stop_reason = 'accepted';
        end

        layer_progress(end+1).pass_idx = pass_idx; %#ok<AGROW>
        layer_progress(end).layer_depth = active_depth;
        layer_progress(end).num_parents_scanned = numel(layer_map(layer_idx).node_ids);
        layer_progress(end).num_candidates = numel(blocks);
        layer_progress(end).num_candidates_evaluated = min( ...
            cfg.parentpair_dev_eval_topk, numel(blocks));
        layer_progress(end).num_blocks_applied = layer_applied;
        if layer_applied >= cfg.max_blocks_per_layer
            stop_reason = 'layer_quota_reached';
        end
        layer_progress(end).stop_reason = stop_reason;
    end

    if ~pass_applied
        break;
    end
end

H1 = current_tree;
end
