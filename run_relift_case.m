function result = run_relift_case(dataset_name, train_file, test_file, seed, val_ratio, fuse_lambda, opts)
%RUN_RELIFT_CASE Run the retained CHA-CF reproduction pipeline.
% The pipeline evaluates the supplied hierarchy, uses it as H0, and applies
% bottom-up local adaptation to obtain H1. Only the AWAphog x HFS-MIMR
% setting documented in README.md is supported by this release.

if nargin < 7 || isempty(opts)
    opts = struct();
end

input_files = {train_file, test_file};
for file_id = 1:numel(input_files)
    if ~isfile(input_files{file_id})
        error('CHACF:MissingInput', 'Required data file not found: %s', input_files{file_id});
    end
end

cfg = local_build_config(opts);

fprintf('=== CHA-CF (%s, MIMR, bottom-up) ===\n', dataset_name);
fprintf('Initial hierarchy: dataset class hierarchy\n');
fprintf('Parameters: rho=%.2f, seed=%d, lambda=%.2f, K_pair=%d, K_dev=%d\n', ...
    val_ratio, seed, fuse_lambda, cfg.parentpair_topk, cfg.parentpair_dev_eval_topk);

train_dat = load(train_file);
test_dat = load(test_file);
train_data = train_dat.data_array;
test_data = test_dat.data_array;
base_tree = train_dat.tree;

% Evaluate the supplied hierarchy as the baseline.
baseline_state = relift_train_state(train_data, base_tree, dataset_name, cfg);
count_num = tabulate(train_data(:, end));
mon_index = find(count_num(:, 2) <= max(count_num(:, 2)) * 0.2);
[baseline_acc, ~, baseline_flca, baseline_fh, baseline_tie_raw, ~, ~, ~, ~, ~, ~, ~, ~] = ...
    HierSVMPredictionBatch( ...
        test_data, base_tree, baseline_state.feature_slct, mon_index, dataset_name, 0);
baseline_tie = baseline_tie_raw / size(test_data, 1);

% The published setting starts adaptation from the dataset hierarchy itself.
H0 = base_tree;
compact_train = train_data;
[fit_data, dev_data] = relift_make_holdout(compact_train, val_ratio, seed);

% Keep the candidate-search random state aligned with the published run.
rng_after_holdout = rng;
initial_state = relift_train_state(compact_train, H0, dataset_name, cfg);
[initial_acc, ~, initial_flca, initial_fh, initial_tie_raw, ~, ~, ~, ~, ~, ~, ~, ~] = ...
    HierSVMPredictionBatch( ...
        test_data, H0, initial_state.feature_slct, mon_index, dataset_name, 0);
initial_tie = initial_tie_raw / size(test_data, 1);
rng(rng_after_holdout);

[H1, applied_blocks, layer_progress] = relift_run_bottomup_repair( ...
    H0, compact_train, fit_data, dev_data, dataset_name, fuse_lambda, cfg);

chacf_state = relift_train_state(compact_train, H1, dataset_name, cfg);
[chacf_acc, ~, chacf_flca, chacf_fh, chacf_tie_raw, ~, ~, ~, ~, ~, ~, ~, ~] = ...
    HierSVMPredictionBatch( ...
        test_data, H1, chacf_state.feature_slct, mon_index, dataset_name, 0);
chacf_tie = chacf_tie_raw / size(test_data, 1);

% Evaluate H0 and H1 on the same development split.
initial_pred = FS_topDownSVMPrediction( ...
    dev_data(:, 1:end-1), initial_state.models, H0, ...
    initial_state.feature_slct, initial_state.numberSel, 0)';
chacf_pred = FS_topDownSVMPrediction( ...
    dev_data(:, 1:end-1), chacf_state.models, H1, ...
    chacf_state.feature_slct, chacf_state.numberSel, 0)';
train_counts = accumarray(compact_train(:, end), 1);
select_initial = relift_eval_leaf_metrics(dev_data(:, end), initial_pred, train_counts);
select_chacf = relift_eval_leaf_metrics(dev_data(:, end), chacf_pred, train_counts);

fprintf('\nApplied blocks:\n');
if isempty(applied_blocks)
    fprintf('none\n\n');
else
    for block_id = 1:numel(applied_blocks)
        block = applied_blocks(block_id);
        fprintf('iter %d pass %d layer_round %d type %s parent %d children ', ...
            block.global_iter, block.pass_idx, block.layer_round, ...
            block.type, block.parent);
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

fprintf('%-12s | %-8s %-8s %-8s %-8s\n', 'Path', 'Acc', 'FH', 'TIE', 'FLCA');
fprintf('%s\n', repmat('-', 1, 54));
fprintf('%-12s | %-8.4f %-8.4f %-8.4f %-8.4f\n', ...
    'baseline', baseline_acc, baseline_fh, baseline_tie, baseline_flca);
fprintf('%-12s | %-8.4f %-8.4f %-8.4f %-8.4f\n', ...
    'initial H0', initial_acc, initial_fh, initial_tie, initial_flca);
fprintf('%-12s | %-8.4f %-8.4f %-8.4f %-8.4f\n', ...
    'CHA-CF', chacf_acc, chacf_fh, chacf_tie, chacf_flca);

result = struct();
result.dataset = dataset_name;
result.feature_selector = 'MIMR';
result.repair_mode = 'bottomup';
result.baseline = struct('acc', baseline_acc, 'fh', baseline_fh, ...
    'tie', baseline_tie, 'flca', baseline_flca);
result.initial = struct('acc', initial_acc, 'fh', initial_fh, ...
    'tie', initial_tie, 'flca', initial_flca);
result.chacf = struct('acc', chacf_acc, 'fh', chacf_fh, ...
    'tie', chacf_tie, 'flca', chacf_flca);
result.H0 = H0;
result.H1 = H1;
result.applied_blocks = applied_blocks;
result.layer_progress = layer_progress;
result.select_initial = select_initial;
result.select_chacf = select_chacf;
end

function cfg = local_build_config(opts)
if isfield(opts, 'fs_method') && ~isempty(opts.fs_method) && ...
        ~strcmpi(char(string(opts.fs_method)), 'MIMR')
    error('CHACF:UnsupportedFeatureSelector', ...
        'This release supports only the MIMR feature selector.');
end
if isfield(opts, 'diag_mode') && ~isempty(opts.diag_mode) && ...
        ~strcmpi(char(string(opts.diag_mode)), 'bottomup')
    error('CHACF:UnsupportedRepairMode', ...
        'This release supports only bottom-up adaptation.');
end
if isfield(opts, 'clf_method') && ~isempty(opts.clf_method) && ...
        ~strcmpi(char(string(opts.clf_method)), 'SVM')
    error('CHACF:UnsupportedClassifier', ...
        'This release supports only the linear SVM classifier.');
end

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
cfg.whole_parent_enable = false;
cfg.whole_parent_min_children = 3;
cfg.whole_parent_max_children = 6;
cfg.whole_parent_max_leaves = 30;
cfg.max_global_passes = 2;
cfg.max_blocks_per_layer = 1;
cfg.accept_min_delta = 0;
cfg.max_repairs_per_parent = 1;
cfg.fs_method = 'MIMR';
cfg.mimr_lambda = 10;
cfg.mimr_alpha = 0.1;
cfg.mimr_beta = 0.01;
cfg.mimr_maxIte = 10;
cfg.mimr_flag = 0;
cfg.feature_select_ratio = [];
cfg.svm_cmd = '-c 1 -t 0 -q';

field_names = fieldnames(cfg);
for field_id = 1:numel(field_names)
    field_name = field_names{field_id};
    if isfield(opts, field_name) && ~isempty(opts.(field_name))
        cfg.(field_name) = opts.(field_name);
    end
end
cfg.fs_method = 'MIMR';
end
