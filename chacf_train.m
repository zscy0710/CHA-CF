function state = chacf_train(data, tree, opt)
%CHACF_TRAIN Train node classifiers for one hierarchy.

[X, Y] = create_SubTable2(data, tree);
feature = chacf_mimr(X, Y, tree, opt.mimrLambda, opt.mimrAlpha, ...
    opt.mimrBeta, opt.mimrIter);

[trainData, trainLabel] = creatSubTablezh(data, tree);
leaf = tree_LeafNode(tree);
models = cell(size(tree, 1), 1);
nFeature = min(opt.featureCount, size(data, 2) - 1);

for node = 1:size(tree, 1)
    if ismember(node, leaf)
        continue;
    end
    selected = feature{node}(1:nFeature);
    models{node} = svmtrain( ...
        trainLabel{node}, trainData{node}(:, selected), opt.svm);
end

state = struct();
state.tree = tree;
state.feature = feature;
state.models = models;
state.nFeature = nFeature;
end
