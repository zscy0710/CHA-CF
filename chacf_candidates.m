function candidates = chacf_candidates(tree, state, devData, fitData, updates, depth, opt)
%CHACF_CANDIDATES Locate and rank confused child pairs at one depth.

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
candidates = candidates(order);
candidates = candidates(1:min(opt.topCandidates, numel(candidates)));
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
    overlap = numel(intersect(leaves, oldLeaves)) / numel(union(leaves, oldLeaves));
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
