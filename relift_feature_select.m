function [feature_slct, W_full, info] = relift_feature_select(train_data, tree, fs_cfg, warm_W)
%RELIFT_FEATURE_SELECT Run the retained MIMR feature-selection method.

if nargin < 3 || isempty(fs_cfg)
    fs_cfg = struct();
end
if nargin < 4
    warm_W = {};
end

if isfield(fs_cfg, 'fs_method') && ~isempty(fs_cfg.fs_method) && ...
        ~strcmpi(char(string(fs_cfg.fs_method)), 'MIMR')
    error('CHACF:UnsupportedFeatureSelector', ...
        'This release supports only the MIMR feature selector (received %s).', ...
        char(string(fs_cfg.fs_method)));
end

lambda = local_option(fs_cfg, 'mimr_lambda', 10);
alpha = local_option(fs_cfg, 'mimr_alpha', 0.1);
beta = local_option(fs_cfg, 'mimr_beta', 0.01);
maxIte = local_option(fs_cfg, 'mimr_maxIte', 10);
flag = local_option(fs_cfg, 'mimr_flag', 0);

opts = struct();
opts.legacy = local_option(fs_cfg, 'mimr_legacy', false);
opts.hoist_gram = local_option(fs_cfg, 'mimr_hoist_gram', true);
opts.skip_empty = local_option(fs_cfg, 'mimr_skip_empty', true);
opts.tol = local_option(fs_cfg, 'mimr_tol', 0);
opts.warm_W = warm_W;

[X, Y] = create_SubTable2(train_data, tree);
[feature_slct, W_full, info] = relift_mimr_feature_select( ...
    X, Y, tree, lambda, alpha, beta, maxIte, flag, opts);
end

function value = local_option(options, name, default_value)
value = default_value;
if isfield(options, name) && ~isempty(options.(name))
    value = options.(name);
end
end
