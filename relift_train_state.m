function [state, W_full] = relift_train_state(train_data, tree, dataset_name, fs_cfg, warm_W)
%RELIFT_TRAIN_STATE Train MIMR feature selectors and linear SVM models.

if nargin < 4 || isempty(fs_cfg)
    fs_cfg = struct();
end
if nargin < 5
    warm_W = {};
end
if isfield(fs_cfg, 'clf_method') && ~isempty(fs_cfg.clf_method) && ...
        ~strcmpi(char(string(fs_cfg.clf_method)), 'SVM')
    error('CHACF:UnsupportedClassifier', ...
        'This release supports only the linear SVM classifier.');
end

if ~isempty(warm_W) || nargout >= 2
    [feature_slct, W_full] = relift_feature_select( ...
        train_data, tree, fs_cfg, warm_W);
else
    feature_slct = relift_feature_select(train_data, tree, fs_cfg);
    W_full = {};
end
num_feature = size(train_data, 2) - 1;
numberSel = relift_select_feature_count(num_feature, dataset_name, fs_cfg);

[trainDataMod, trainLabelMod] = creatSubTablezh(train_data, tree);
leaf_nodes = tree_LeafNode(tree);
models = cell(size(tree, 1), 1);

if isfield(fs_cfg, 'svm_cmd') && ~isempty(fs_cfg.svm_cmd)
    svm_cmd = fs_cfg.svm_cmd;
else
    svm_cmd = '-c 1 -t 0 -q';
end

for node_id = 1:size(tree, 1)
    if ismember(node_id, leaf_nodes)
        continue;
    end
    train_labels = trainLabelMod{node_id};
    train_features = trainDataMod{node_id};
    if isempty(train_labels) || numel(unique(train_labels)) < 2
        error('CHACF:UntrainableInternalNode', ...
            'Internal node %d has no usable two-child training task.', node_id);
    end
    selected = feature_slct{node_id}( ...
        1:min(numberSel, numel(feature_slct{node_id})));
    models{node_id} = svmtrain( ...
        train_labels, train_features(:, selected), svm_cmd);
end

state = struct();
state.tree = tree;
state.feature_slct = feature_slct;
state.models = models;
state.numberSel = numberSel;
state.clf_method = "svm";
state.scaler = [];
end
