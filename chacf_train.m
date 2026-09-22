function state = chacf_train(data, tree, opt)
[X, Y] = create_SubTable2(data, tree);
feature = chacf_mimr(X, Y, tree, opt.mimrLambda, opt.mimrAlpha, ...
    opt.mimrBeta, opt.mimrIter);
[trainData, trainLabel] = creatSubTablezh(data, tree);
leaf = tree_LeafNode(tree);
models = cell(size(tree, 1), 1);
nFeature = min(opt.featureCount, size(data, 2) - 1);
for node = 1:size(tree, 1)
    if ~ismember(node, leaf)
        selected = feature{node}(1:nFeature);
        models{node} = svmtrain( ...
            trainLabel{node}, trainData{node}(:, selected), opt.svm);
    end
end
state = struct();
state.tree = tree;
state.feature = feature;
state.models = models;
state.nFeature = nFeature;
end

function feature = chacf_mimr(X, Y, tree, lambda, alpha, beta, maxIter)
internal = tree_InternalNodes(tree);
internal(internal == -1) = [];
root = tree_Root(tree);
nodes = [internal; root];
numSelected = size(X{root}, 2);
numClass = zeros(max(nodes), 1);
for i = 1:numel(nodes)
    numClass(nodes(i)) = numel(unique(Y{nodes(i)}));
end
maxClass = max(numClass);

leaf = tree_LeafNode(tree);
numSample = zeros(max(nodes), 1);
for i = 1:numel(nodes)
    numSample(nodes(i)) = size(X{nodes(i)}, 1);
end
empty = nodes(numSample(nodes) == 0);
internal = setdiff(internal, empty, 'stable');
solveNodes = [internal; root];
exclude = [leaf(:); empty(:)];

W = cell(max(nodes), 1);
XX = cell(max(nodes), 1);
XY = cell(max(nodes), 1);
D = cell(max(nodes), 1);
for i = 1:numel(nodes)
    node = nodes(i);
    d = size(X{node}, 2);
    Y{node} = conversionY01_extend(Y{node}, maxClass);
    W{node} = ones(d, maxClass);
end
for i = 1:numel(solveNodes)
    node = solveNodes(i);
    XX{node} = X{node}' * X{node};
    XY{node} = X{node}' * Y{node};
end

for iter = 1:maxIter
    for i = 1:numel(solveNodes)
        node = solveNodes(i);
        D{node} = diag(0.5 ./ max(sqrt(sum(W{node}.^2, 2)), eps));
    end
    d = size(X{root}, 2);
    W{root} = (XX{root} + lambda * D{root} + ...
        beta * (ones(d) - eye(d))) \ XY{root};
    for i = 1:numel(internal)
        node = internal(i);
        d = size(X{node}, 2);
        U1 = zeros(d);
        U2 = zeros(d, maxClass);
        siblings = setdiff(tree_Sibling(tree, node), exclude);
        for j = 1:numel(siblings)
            sibling = siblings(j);
            U1 = U1 + W{sibling} * W{sibling}';
            U2 = U2 + W{sibling};
        end
        W{node} = (XX{node} + lambda * D{node} + ...
            beta * (ones(d) - eye(d)) + alpha * U1) \ ...
            (XY{node} + alpha * U2);
    end
end

feature = cell(max(nodes), 1);
for i = 1:numel(nodes)
    node = nodes(i);
    score = sum(W{node}(:, 1:numClass(node)).^2, 2);
    [~, order] = sort(score, 'descend');
    feature{node} = order(1:numSelected);
end
end
