function state = relift_train_state(train_data, tree, dataset_name, cfg)
%RELIFT_TRAIN_STATE Train MIMR feature selectors and linear SVM models.

feature_slct = relift_feature_select(train_data, tree, cfg);
num_feature = size(train_data, 2) - 1;
numberSel = relift_select_feature_count(num_feature, dataset_name, cfg);

[trainDataMod, trainLabelMod] = creatSubTablezh(train_data, tree);
leaf_nodes = tree_LeafNode(tree);
models = cell(size(tree, 1), 1);

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
        train_labels, train_features(:, selected), cfg.svm_cmd);
end

state = struct();
state.tree = tree;
state.feature_slct = feature_slct;
state.models = models;
state.numberSel = numberSel;
end
