function result = run_awa_mimr()
%RUN_AWA_MIMR Reproduce the CHA-CF AWAphog x HFS-MIMR result.
% Add the CHA-CF project root to the MATLAB path, then call:
%   result = run_awa_mimr();
%
% This is the retained runnable setting:
%   AWAphog -> MIMR feature selection -> supplied hierarchy H0
%   -> bottom-up local hierarchy adaptation H1.
%
% Published parameters:
%   rho = 0.25, lambda = 0.30, seed = 13, K_pair = 8, K_dev = 5,
%   and a linear SVM with '-c 1 -t 0 -q'.
%
% Expected results, rounded to four decimals:
%   baseline  Acc 0.2449  F_H 0.5737  TIE 0.3410
%   CHA-CF    Acc 0.2664  F_H 0.7327  TIE 0.1552

project_root = fileparts(mfilename('fullpath'));
train_file = fullfile(project_root, 'datasets', 'AWAphogTrain.mat');
test_file = fullfile(project_root, 'datasets', 'AWAphogTest.mat');

if ~isfile(train_file) || ~isfile(test_file)
    error('CHACF:MissingAWAData', ...
        'AWAphog data is incomplete. Expected %s and %s.', train_file, test_file);
end

opts = struct( ...
    'fs_method', 'MIMR', ...
    'max_global_passes', 2, ...
    'max_blocks_per_layer', 1, ...
    'accept_min_delta', 0, ...
    'max_repairs_per_parent', 1, ...
    'whole_parent_enable', false, ...
    'parentpair_pairs_per_parent', 2, ...
    'parentpair_topk', 8, ...
    'parentpair_dev_eval_topk', 5);

result = run_relift_case( ...
    'AWAphog', train_file, test_file, 13, 0.25, 0.30, opts);
end
