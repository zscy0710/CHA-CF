function [H1, updates] = chacf_adapt(H0, trainData, fitData, devData, opt)
H1 = H0;
updates = struct('pass', {}, 'depth', {}, 'parent', {}, ...
    'children', {}, 'leaves', {}, 'key', {}, 'score', {});

for pass = 1:opt.maxPasses
    changed = false;
    nodes = [tree_InternalNodes(H1); tree_Root(H1)];
    depths = sort(unique(H1(nodes, 2)), 'descend');
    for i = 1:numel(depths)
        depth = depths(i);
        nUpdate = 0;
        while nUpdate < opt.maxUpdatesPerLevel
            state = chacf_train(fitData, H1, opt);
            candidates = chacf_candidates( ...
                H1, state, devData, fitData, updates, depth, opt);
            if isempty(candidates)
                break;
            end
            [choice, Hnext, accepted] = chacf_select( ...
                H1, trainData, fitData, devData, state, candidates, opt);
            if ~accepted
                break;
            end
            H1 = Hnext;
            updates(end+1) = struct( ...
                'pass', pass, 'depth', depth, 'parent', choice.parent, ...
                'children', choice.children, 'leaves', choice.leaves, ...
                'key', choice.key, 'score', choice.score);
            nUpdate = nUpdate + 1;
            changed = true;
        end
    end
    if ~changed
        break;
    end
end
end

function candidates = chacf_candidates(tree, state, devData, fitData, updates, depth, opt)
nodes = [tree_InternalNodes(tree); tree_Root(tree)];
nodes = nodes(tree(nodes, 2) == depth);
candidates = struct('parent', {}, 'children', {}, 'leaves', {}, ...
    'key', {}, 'mass', {}, 'support', {}, 'overlap', {}, 'score', {});

for i = 1:numel(nodes)
    parent = nodes(i);
    leaves = chacf_descendants(tree, parent);
    key = chacf_signature(tree, parent);
    if local_count_updates(updates, key) >= opt.maxRepeat
        continue;
    end
    support = sum(ismember(devData(:, end), leaves));
    if support < opt.minSupport
        continue;
    end
    [confusion, children, totalWeight] = chacf_parent_confusion( ...
        tree, parent, state, devData, fitData);
    if totalWeight == 0
        continue;
    end
    pairs = local_top_pairs(tree, children, confusion, totalWeight, opt.pairsPerNode);
    for j = 1:numel(pairs)
        [overlap, seen] = local_overlap( ...
            key, pairs(j).children, pairs(j).leaves, updates, opt);
        if seen
            continue;
        end
        candidates(end+1) = struct( ...
            'parent', parent, 'children', pairs(j).children, ...
            'leaves', pairs(j).leaves, 'key', key, ...
            'mass', pairs(j).mass, ...
            'support', min(1, support / opt.supportFloor), ...
            'overlap', overlap, 'score', 0);
    end
end

if isempty(candidates)
    return;
end
maxMass = max([candidates.mass]);
for i = 1:numel(candidates)
    candidates(i).score = candidates(i).mass / maxMass * ...
        candidates(i).support * candidates(i).overlap;
end
candidates = candidates([candidates.score] >= opt.minScore);
if isempty(candidates)
    return;
end
[~, order] = sort([candidates.score], 'descend');
candidates = candidates(order(1:min(opt.topCandidates, numel(order))));
end

function n = local_count_updates(updates, key)
n = 0;
for i = 1:numel(updates)
    n = n + strcmp(updates(i).key, key);
end
end

function [factor, seen] = local_overlap(key, children, leaves, updates, opt)
factor = 1;
seen = false;
maxOverlap = 0;
children = sort(children(:))';
leaves = unique(leaves(:)');
for i = 1:numel(updates)
    if ~strcmp(updates(i).key, key)
        continue;
    end
    oldChildren = sort(updates(i).children(:))';
    if isequal(children, oldChildren)
        seen = true;
        return;
    end
    oldLeaves = unique(updates(i).leaves(:)');
    overlap = numel(intersect(leaves, oldLeaves)) / ...
        numel(union(leaves, oldLeaves));
    if overlap >= opt.maxLeafOverlap
        seen = true;
        return;
    end
    maxOverlap = max(maxOverlap, overlap);
end
if maxOverlap >= opt.overlapThreshold
    factor = opt.overlapPenalty;
else
    factor = 1 - maxOverlap;
end
factor = max(factor, eps);
end

function pairs = local_top_pairs(tree, children, confusion, totalWeight, maxPairs)
pairs = struct('children', {}, 'leaves', {}, 'mass', {});
pairMass = confusion + confusion';
pairMass(1:size(pairMass, 1)+1:end) = 0;
index = [];
for i = 1:numel(children)
    for j = i+1:numel(children)
        if pairMass(i, j) > 0
            index(end+1, :) = [i, j, pairMass(i, j)];
        end
    end
end
if isempty(index)
    return;
end
[~, order] = sort(index(:, 3), 'descend');
index = index(order(1:min(maxPairs, numel(order))), :);
for i = 1:size(index, 1)
    pairChildren = children(index(i, 1:2));
    pairLeaves = [];
    for j = 1:numel(pairChildren)
        pairLeaves = [pairLeaves; chacf_descendants(tree, pairChildren(j))];
    end
    pairs(end+1) = struct( ...
        'children', pairChildren(:)', 'leaves', unique(pairLeaves(:)'), ...
        'mass', index(i, 3) / totalWeight);
end
end

function [choice, Hnext, accepted] = chacf_select( ...
    tree, trainData, fitData, devData, state, candidates, opt)
baseScore = local_score(tree, state, devData);
bestScore = -Inf;
choice = candidates(1);
Hnext = tree;
for i = 1:min(opt.topEval, numel(candidates))
    trialTree = chacf_rebuild( ...
        tree, trainData, candidates(i), state, devData, fitData, opt.fusion);
    trialState = chacf_train(fitData, trialTree, opt);
    score = local_score(trialTree, trialState, devData);
    if score > bestScore
        bestScore = score;
        choice = candidates(i);
        Hnext = trialTree;
    end
end
accepted = bestScore > baseScore;
if ~accepted
    Hnext = tree;
end
end

function score = local_score(tree, state, devData)
pred = FS_topDownSVMPrediction( ...
    devData(:, 1:end-1), state.models, tree, state.feature, state.nFeature, 0)';
pred = double(pred(:));
[fh, tie] = chacf_local_metrics(tree, tree_Root(tree), devData(:, end), pred);
score = mean(devData(:, end) == pred) + fh - tie;
end

function newTree = chacf_rebuild(tree, trainData, candidate, state, devData, fitData, fusion)
A0 = chacf_feature_affinity(trainData, candidate.leaves);
A1 = chacf_confusion_affinity(candidate, state, devData, fitData);
groups = chacf_cluster(fusion * A0 + (1 - fusion) * A1, ...
    candidate.leaves, numel(candidate.children));
spec = chacf_tree_to_spec(tree, tree_Root(tree));
spec = chacf_replace_block(spec, candidate.parent, candidate.children, groups);
newTree = chacf_spec_to_tree(spec);
end

function affinity = chacf_feature_affinity(data, labels)
labels = labels(:)';
n = numel(labels);
affinity = zeros(n);
count = zeros(n, 1);
prototype = zeros(n, size(data, 2) - 1);
for i = 1:n
    x = data(data(:, end) == labels(i), 1:end-1);
    count(i) = size(x, 1);
    prototype(i, :) = mean(x, 1);
end
for i = 1:n
    affinity(i, i) = 1;
    for j = i+1:n
        distance = 1 / (1 + norm(prototype(i, :) - prototype(j, :)));
        correlation = corr(prototype(i, :)', prototype(j, :)', 'type', 'Pearson');
        if isnan(correlation)
            correlation = 0;
        end
        balance = 1 / (1 + abs(count(i) - count(j)));
        affinity(i, j) = max(correlation, 0) * distance * balance / ...
            (count(i) + count(j));
        affinity(j, i) = affinity(i, j);
    end
end
end

function affinity = chacf_confusion_affinity(candidate, state, devData, fitData)
labels = candidate.leaves(:)';
affinity = zeros(numel(labels));
index = ismember(devData(:, end), chacf_descendants(state.tree, candidate.parent));
if ~any(index)
    return;
end
pred = chacf_predict(devData(index, 1:end-1), state, candidate.parent);
truth = devData(index, end);
classCount = accumarray(fitData(:, end), 1, [max(fitData(:, end)), 1]);
weight = classCount(truth) .^ -1;
for i = 1:numel(truth)
    r = find(labels == truth(i), 1);
    c = find(labels == pred(i), 1);
    if ~isempty(r) && ~isempty(c) && r ~= c
        affinity(r, c) = affinity(r, c) + weight(i);
    end
end
for i = 1:numel(labels)
    rowSum = sum(affinity(i, :));
    if rowSum > 0
        affinity(i, :) = affinity(i, :) / rowSum;
    end
end
affinity = (affinity + affinity') / 2;
affinity(1:size(affinity, 1)+1:end) = 0;
end

function groups = chacf_cluster(affinity, labels, nGroup)
n = size(affinity, 1);
if n == 1
    groups = {labels};
    return;
end
nGroup = min(nGroup, n);
L = diag(sum(affinity, 2)) - affinity;
[V, D] = eig((L + L') / 2);
[~, order] = sort(diag(D), 'ascend');
cluster = kmeans(V(:, order(1:nGroup)), nGroup, ...
    'Replicates', 20, 'Display', 'off');
groups = cell(1, nGroup);
for i = 1:nGroup
    groups{i} = labels(cluster == i)';
end
groups = groups(~cellfun(@isempty, groups));
end

function [confusion, children, totalWeight] = chacf_parent_confusion( ...
    tree, parent, state, devData, fitData)
children = get_children_set(tree, parent);
leaves = chacf_descendants(tree, parent);
index = ismember(devData(:, end), leaves);
confusion = zeros(numel(children));
totalWeight = 0;
if numel(children) < 2 || ~any(index)
    return;
end
pred = chacf_predict(devData(index, 1:end-1), state, parent);
truth = devData(index, end);
classCount = accumarray(fitData(:, end), 1, [max(fitData(:, end)), 1]);
weight = classCount(truth) .^ -1;
totalWeight = sum(weight);
for i = 1:numel(truth)
    trueChild = chacf_child_on_path(tree, parent, truth(i));
    predChild = chacf_child_on_path(tree, parent, pred(i));
    r = find(children == trueChild, 1);
    c = find(children == predChild, 1);
    if ~isempty(r) && ~isempty(c) && r ~= c
        confusion(r, c) = confusion(r, c) + weight(i);
    end
end
end

function pred = chacf_predict(data, state, startNode)
pred = zeros(size(data, 1), 1);
leaf = tree_LeafNode(state.tree);
for i = 1:size(data, 1)
    node = startNode;
    while ~ismember(node, leaf)
        node = svmpredict(0, data(i, state.feature{node}(1:state.nFeature)), ...
            state.models{node}, '-q');
    end
    pred(i) = node;
end
end

function [fh, tie] = chacf_local_metrics(tree, node, truth, pred)
leaves = chacf_descendants(tree, node);
index = ismember(truth, leaves);
if ~any(index)
    fh = 0;
    tie = 1;
    return;
end
truth = truth(index);
pred = pred(index);
maxDistance = local_max_distance(tree, node);
sumFH = 0;
sumTIE = 0;
for i = 1:numel(truth)
    truePath = local_path(tree, node, truth(i));
    predPath = local_path(tree, node, pred(i));
    common = intersect(truePath, predPath);
    p = numel(common) / numel(predPath);
    r = numel(common) / numel(truePath);
    if p + r > 0
        sumFH = sumFH + 2 * p * r / (p + r);
    end
    sumTIE = sumTIE + numel(setxor(truePath, predPath));
end
fh = sumFH / numel(truth);
tie = sumTIE / numel(truth) / maxDistance;
end

function path = local_path(tree, node, leaf)
path = leaf;
while path(end) ~= node && path(end) ~= 0
    parent = tree(path(end), 1);
    path(end+1) = parent;
end
path(path == 0) = [];
end

function d = local_max_distance(tree, node)
leaves = chacf_descendants(tree, node);
d = 1;
for i = 1:numel(leaves)
    for j = i+1:numel(leaves)
        d = max(d, numel(setxor( ...
            local_path(tree, node, leaves(i)), local_path(tree, node, leaves(j)))));
    end
end
end

function leaves = chacf_descendants(tree, node)
children = get_children_set(tree, node);
if isempty(children)
    leaves = node;
    return;
end
leaves = [];
for i = 1:numel(children)
    leaves = [leaves; chacf_descendants(tree, children(i))];
end
end

function child = chacf_child_on_path(tree, parent, leaf)
child = leaf;
while tree(child, 1) ~= parent && tree(child, 1) ~= 0
    child = tree(child, 1);
end
end

function key = chacf_signature(tree, node)
leaves = sort(unique(chacf_descendants(tree, node)));
key = sprintf('%d_', leaves);
key(end) = [];
end

function spec = chacf_tree_to_spec(tree, node)
if ismember(node, tree_LeafNode(tree))
    spec = node;
    return;
end
children = get_children_set(tree, node);
subtree = cell(1, numel(children));
for i = 1:numel(children)
    subtree{i} = chacf_tree_to_spec(tree, children(i));
end
spec = struct('id', node, 'children', {subtree});
end

function spec = chacf_replace_block(spec, parent, oldChildren, groups)
if isnumeric(spec)
    return;
end
if ~isempty(spec.id) && spec.id == parent
    kept = {};
    for i = 1:numel(spec.children)
        child = spec.children{i};
        if ~ismember(local_root(child), oldChildren)
            kept{end+1} = child;
        end
    end
    for i = 1:numel(groups)
        if numel(groups{i}) == 1
            kept{end+1} = groups{i};
        else
            kept{end+1} = struct('id', [], ...
                'children', {num2cell(groups{i})});
        end
    end
    spec.children = kept;
    return;
end
for i = 1:numel(spec.children)
    spec.children{i} = chacf_replace_block( ...
        spec.children{i}, parent, oldChildren, groups);
end
end

function id = local_root(spec)
if isnumeric(spec)
    id = spec;
else
    id = spec.id;
end
end

function tree = chacf_spec_to_tree(spec)
leaves = sort(unique(local_leaves(spec)));
nLeaf = max(leaves);
parent = zeros(4 * nLeaf, 1);
next = nLeaf + 1;
root = local_emit(spec, 0);
parent(root) = 0;
parent = parent(1:next-1);
tree = [parent, local_level(parent)];

    function node = local_emit(part, parentNode)
        if isnumeric(part)
            node = part;
            parent(node) = parentNode;
            return;
        end
        node = next;
        next = next + 1;
        if node > numel(parent)
            parent = [parent; zeros(nLeaf, 1)];
        end
        parent(node) = parentNode;
        for k = 1:numel(part.children)
            local_emit(part.children{k}, node);
        end
    end
end

function leaves = local_leaves(spec)
if isnumeric(spec)
    leaves = spec;
    return;
end
leaves = [];
for i = 1:numel(spec.children)
    leaves = [leaves, local_leaves(spec.children{i})];
end
end

function level = local_level(parent)
level = zeros(size(parent));
nodes = find(parent == 0, 1);
depth = 0;
while ~isempty(nodes)
    nodes = find(ismember(parent, nodes));
    depth = depth + 1;
    level(nodes) = depth;
end
end
