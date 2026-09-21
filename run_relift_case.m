function result = run_relift_case(dataset_name, train_file, test_file, seed, val_ratio, fuse_lambda)
%RUN_RELIFT_CASE Run one CHA-CF experiment from the supplied hierarchy.

cfg = local_config();

fprintf('=== CHA-CF (%s, MIMR, bottom-up) ===\n', dataset_name);
fprintf('Initial hierarchy: dataset class hierarchy\n');
fprintf('Parameters: rho=%.2f, seed=%d, lambda=%.2f, K_pair=%d, K_dev=%d\n', ...
    val_ratio, seed, fuse_lambda, cfg.parentpair_topk, cfg.parentpair_dev_eval_topk);

train_dat = load(train_file);
test_dat = load(test_file);
train_data = train_dat.data_array;
test_data = test_dat.data_array;
H0 = train_dat.tree;

baseline_state = relift_train_state(train_data, H0, dataset_name, cfg);
count_num = tabulate(train_data(:, end));
mon_index = find(count_num(:, 2) <= max(count_num(:, 2)) * 0.2);
[baseline_acc, ~, baseline_flca, baseline_fh, baseline_tie_raw, ~, ~, ~, ~, ~, ~, ~, ~] = ...
    HierSVMPredictionBatch( ...
    test_data, H0, baseline_state.feature_slct, mon_index, dataset_name, 0);
baseline_tie = baseline_tie_raw / size(test_data, 1);

[fit_data, dev_data] = relift_make_holdout(train_data, val_ratio, seed);
[H1, applied_blocks, layer_progress] = relift_run_bottomup_repair( ...
    H0, train_data, fit_data, dev_data, dataset_name, fuse_lambda, cfg);

chacf_state = relift_train_state(train_data, H1, dataset_name, cfg);
[chacf_acc, ~, chacf_flca, chacf_fh, chacf_tie_raw, ~, ~, ~, ~, ~, ~, ~, ~] = ...
    HierSVMPredictionBatch( ...
    test_data, H1, chacf_state.feature_slct, mon_index, dataset_name, 0);
chacf_tie = chacf_tie_raw / size(test_data, 1);

fprintf('\nApplied blocks:\n');
if isempty(applied_blocks)
    fprintf('none\n\n');
else
    for block_id = 1:numel(applied_blocks)
        block = applied_blocks(block_id);
        fprintf('iter %d pass %d layer_round %d parent %d children ', ...
            block.global_iter, block.pass_idx, block.layer_round, block.parent);
        fprintf('%d ', block.pair_children);
        fprintf('| score %.4f | leaves ', block.score);
        fprintf('%d ', block.pair_leaf_labels);
        fprintf('\n');
    end
    fprintf('\n');
end

if ~isempty(layer_progress)
    fprintf('Layer progress:\n');
    for layer_id = 1:numel(layer_progress)
        layer = layer_progress(layer_id);
        fprintf('pass %d depth %d scanned %d candidates %d eval %d applied %d stop %s\n', ...
            layer.pass_idx, layer.layer_depth, layer.num_parents_scanned, ...
            layer.num_candidates, layer.num_candidates_evaluated, ...
            layer.num_blocks_applied, layer.stop_reason);
    end
    fprintf('\n');
end

fprintf('%-12s | %-8s %-8s %-8s %-8s\n', 'Method', 'Acc', 'FH', 'TIE', 'FLCA');
fprintf('%s\n', repmat('-', 1, 54));
fprintf('%-12s | %-8.4f %-8.4f %-8.4f %-8.4f\n', ...
    'HFS-MIMR', baseline_acc, baseline_fh, baseline_tie, baseline_flca);
fprintf('%-12s | %-8.4f %-8.4f %-8.4f %-8.4f\n', ...
    'CHA-CF', chacf_acc, chacf_fh, chacf_tie, chacf_flca);

result = struct();
result.dataset = dataset_name;
result.baseline = struct('acc', baseline_acc, 'fh', baseline_fh, ...
    'tie', baseline_tie, 'flca', baseline_flca);
result.chacf = struct('acc', chacf_acc, 'fh', chacf_fh, ...
    'tie', chacf_tie, 'flca', chacf_flca);
result.H0 = H0;
result.H1 = H1;
result.applied_blocks = applied_blocks;
result.layer_progress = layer_progress;
end

function cfg = local_config()
cfg = struct();
cfg.min_support = 30;
cfg.support_floor = 100;
cfg.parentpair_topk = 8;
cfg.parentpair_dev_eval_topk = 5;
cfg.parentpair_min_score = 0.12;
cfg.parentpair_min_conf_mass = 0;
cfg.parentpair_overlap_threshold = 0.70;
cfg.parentpair_overlap_penalty = 0.10;
cfg.parentpair_skip_exact = true;
cfg.parentpair_skip_leaf_overlap = 0.95;
cfg.parentpair_pairs_per_parent = 2;
cfg.max_global_passes = 2;
cfg.max_blocks_per_layer = 1;
cfg.accept_min_delta = 0;
cfg.max_repairs_per_parent = 1;
cfg.mimr_lambda = 10;
cfg.mimr_alpha = 0.1;
cfg.mimr_beta = 0.01;
cfg.mimr_maxIte = 10;
cfg.mimr_flag = 0;
cfg.feature_select_ratio = [];
cfg.svm_cmd = '-c 1 -t 0 -q';
end
