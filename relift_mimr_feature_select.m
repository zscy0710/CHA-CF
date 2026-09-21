function feature_slct = relift_mimr_feature_select(X, Y, tree, lambda, alpha, beta, maxIte, flag)
%RELIFT_MIMR_FEATURE_SELECT MIMR feature selection for the current hierarchy.

internalNodes = tree_InternalNodes(tree);
internalNodes(internalNodes == -1) = [];
indexRoot = tree_Root(tree);
noLeafNode = [internalNodes; indexRoot];
numSelected = size(X{indexRoot}, 2);
m = zeros(max(noLeafNode), 1);

for i = 1:numel(noLeafNode)
    node_id = noLeafNode(i);
    m(node_id) = numel(unique(Y{node_id}));
end
maxm = max(m);

leafNode = tree_LeafNode(tree);
nu = zeros(max(noLeafNode), 1);
for i = 1:numel(noLeafNode)
    node_id = noLeafNode(i);
    nu(node_id) = size(X{node_id}, 1);
end
emptyNodes = noLeafNode(nu(noLeafNode) == 0);
if nu(indexRoot) == 0
    error('CoHReHC:MimrEmptyRoot', 'The root node has no training samples.');
end

% Empty internal nodes are excluded from local optimization and sibling coupling.
solveInternal = setdiff(internalNodes, emptyNodes, 'stable');
siblingExclude = [leafNode(:); emptyNodes(:)];
solveNodes = [solveInternal; indexRoot];

W = cell(max(noLeafNode), 1);
XX = cell(max(noLeafNode), 1);
XY = cell(max(noLeafNode), 1);
D = cell(max(noLeafNode), 1);
for i = 1:numel(noLeafNode)
    node_id = noLeafNode(i);
    d = size(X{node_id}, 2);
    Y{node_id} = conversionY01_extend(Y{node_id}, maxm);
    W{node_id} = ones(d, maxm);
end

% These Gram matrices do not change across MIMR iterations.
for i = 1:numel(solveNodes)
    node_id = solveNodes(i);
    XX{node_id} = X{node_id}' * X{node_id};
    XY{node_id} = X{node_id}' * Y{node_id};
end

if flag == 1
    obj = zeros(1, maxIte);
end
for iter = 1:maxIte
    for i = 1:numel(solveNodes)
        node_id = solveNodes(i);
        D{node_id} = diag(0.5 ./ max( ...
            sqrt(sum(W{node_id} .* W{node_id}, 2)), eps));
    end

    d = size(X{indexRoot}, 2);
    W{indexRoot} = (XX{indexRoot} + lambda * D{indexRoot} + ...
        beta * ones(d) - beta * eye(d)) \ XY{indexRoot};

    for i = 1:numel(solveInternal)
        node_id = solveInternal(i);
        d = size(X{node_id}, 2);
        U1 = zeros(d, d);
        U2 = zeros(d, maxm);
        siblingNodes = setdiff(tree_Sibling(tree, node_id), siblingExclude);
        for j = 1:numel(siblingNodes)
            sibling_id = siblingNodes(j);
            U1 = U1 + W{sibling_id} * W{sibling_id}';
            U2 = U2 + W{sibling_id};
        end
        W{node_id} = (XX{node_id} + lambda * D{node_id} + ...
            beta * (ones(d) - eye(d)) + alpha * U1) \ ...
            (XY{node_id} + alpha * U2);
    end

    if flag == 1
        obj(iter) = 1/2 * norm(X{indexRoot} * W{indexRoot} - Y{indexRoot})^2 + ...
            lambda * L21(W{indexRoot}) + ...
            beta * trace(ones(d) * W{indexRoot} * W{indexRoot}' - ...
            W{indexRoot} * W{indexRoot}');
        for i = 1:numel(solveInternal)
            node_id = solveInternal(i);
            siblingNodes = setdiff(tree_Sibling(tree, node_id), siblingExclude);
            S = 0;
            for j = 1:numel(siblingNodes)
                S = S + norm(W{node_id}' * W{siblingNodes(j)} - eye(maxm), 'fro')^2;
            end
            obj(iter) = obj(iter) + 1/2 * norm( ...
                X{node_id} * W{node_id} - Y{node_id})^2 + ...
                lambda/2 * L21(W{node_id}) + ...
                beta * trace(ones(d) * W{node_id} * W{node_id}' - ...
                W{node_id} * W{node_id}') + alpha * S;
        end
    end
end

feature_slct = cell(max(noLeafNode), 1);
for i = 1:numel(noLeafNode)
    node_id = noLeafNode(i);
    node_weights = W{node_id}(:, 1:m(node_id));
    [~, order] = sort(sum(node_weights.^2, 2), 'descend');
    feature_slct{node_id} = order(1:numSelected);
end

if flag == 1
    figure('Color', [1 1 1]);
    plot(obj, 'LineWidth', 4, 'Color', [0 0 1]);
    xlabel('Iteration number');
    ylabel('Objective function value');
end
end
