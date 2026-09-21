function blocks = relift_collect_parentpair_candidates(tree, fit_state, dev_data, fit_data, cfg, applied_blocks, target_level)
%RELIFT_COLLECT_PARENTPAIR_CANDIDATES Generate and rank local child-pair candidates.

internal_nodes = [tree_InternalNodes(tree); tree_Root(tree)];
internal_nodes = internal_nodes(tree(internal_nodes, 2) == target_level);
blocks = struct('parent', {}, 'pair_children', {}, 'pair_leaf_labels', {}, ...
    'parent_signature', {}, 'parent_leaf_labels', {}, 'layer_depth', {}, ...
    'score', {}, 'conf_mass', {}, 'support', {}, 'support_rel', {}, ...
    'max_overlap', {}, 'overlap_rel', {});

for i = 1:numel(internal_nodes)
    node_id = internal_nodes(i);
    leaf_ids = relift_leaf_descendants(tree, node_id);
    parent_signature = relift_node_signature(tree, node_id);
    if relift_parent_repair_count(applied_blocks, parent_signature) >= cfg.max_repairs_per_parent
        continue;
    end
    support = sum(ismember(dev_data(:, end), leaf_ids));
    if support < cfg.min_support
        continue;
    end

    [raw_conf, children, total_weight] = relift_parent_confusion( ...
        tree, node_id, fit_state, dev_data, fit_data);
    if total_weight <= 0 || isempty(children)
        continue;
    end

    pair_table = relift_extract_top_pairs( ...
        tree, children, raw_conf, total_weight, cfg.parentpair_pairs_per_parent);
    for pair_id = 1:numel(pair_table)
        pair_children = pair_table(pair_id).pair_children;
        pair_leaf_labels = pair_table(pair_id).pair_leaf_labels;
        conf_mass = pair_table(pair_id).conf_mass;
        if conf_mass <= cfg.parentpair_min_conf_mass
            continue;
        end

        [overlap_rel, max_overlap, skip_pair] = relift_parentpair_overlap( ...
            parent_signature, pair_children, pair_leaf_labels, applied_blocks, cfg);
        if skip_pair
            continue;
        end

        blocks(end+1).parent = node_id; %#ok<AGROW>
        blocks(end).pair_children = pair_children(:)';
        blocks(end).pair_leaf_labels = pair_leaf_labels(:)';
        blocks(end).parent_signature = parent_signature;
        blocks(end).parent_leaf_labels = leaf_ids(:)';
        blocks(end).layer_depth = tree(node_id, 2);
        blocks(end).conf_mass = conf_mass;
        blocks(end).support = support;
        blocks(end).support_rel = min(1, support / cfg.support_floor);
        blocks(end).max_overlap = max_overlap;
        blocks(end).overlap_rel = overlap_rel;
        blocks(end).score = 0;
    end
end

if isempty(blocks)
    return;
end

max_conf = max([blocks.conf_mass]);
for i = 1:numel(blocks)
    conf_norm = blocks(i).conf_mass / max(max_conf, eps);
    blocks(i).score = conf_norm * blocks(i).support_rel * blocks(i).overlap_rel;
end

blocks = blocks([blocks.score] >= cfg.parentpair_min_score);
if isempty(blocks)
    return;
end

[~, order] = sort([blocks.score], 'descend');
blocks = blocks(order);
blocks = blocks(1:min(cfg.parentpair_topk, numel(blocks)));
end

function [overlap_rel, max_overlap, skip_pair] = relift_parentpair_overlap(parent_signature, pair_children, pair_leaf_labels, applied_blocks, cfg)
skip_pair = false;
max_overlap = 0;
overlap_rel = 1;

pair_children = sort(pair_children(:)).';
pair_leaf_labels = unique(pair_leaf_labels(:)');
for i = 1:numel(applied_blocks)
    if ~strcmp(applied_blocks(i).parent_signature, parent_signature)
        continue;
    end
    past_children = sort(applied_blocks(i).pair_children(:)).';
    if cfg.parentpair_skip_exact && isequal(pair_children, past_children)
        skip_pair = true;
        overlap_rel = 0;
        return;
    end

    past_leaf_labels = unique(applied_blocks(i).pair_leaf_labels(:)');
    union_count = numel(union(pair_leaf_labels, past_leaf_labels));
    if union_count == 0
        continue;
    end
    overlap = numel(intersect(pair_leaf_labels, past_leaf_labels)) / union_count;
    if overlap >= cfg.parentpair_skip_leaf_overlap
        skip_pair = true;
        overlap_rel = 0;
        return;
    end
    max_overlap = max(max_overlap, overlap);
end

if max_overlap >= cfg.parentpair_overlap_threshold
    overlap_rel = cfg.parentpair_overlap_penalty;
else
    overlap_rel = 1 - max_overlap;
end
overlap_rel = max(overlap_rel, eps);
end

function n = relift_parent_repair_count(applied_blocks, parent_signature)
n = 0;
for i = 1:numel(applied_blocks)
    if strcmp(applied_blocks(i).parent_signature, parent_signature)
        n = n + 1;
    end
end
end

function pair_table = relift_extract_top_pairs(tree, children, raw_conf, total_weight, max_pairs)
pair_table = struct('pair_children', {}, 'pair_leaf_labels', {}, 'conf_mass', {});
pair_mass = raw_conf + raw_conf';
pair_mass(1:size(pair_mass, 1)+1:end) = 0;

pair_list = [];
for i = 1:numel(children)
    for j = i+1:numel(children)
        if pair_mass(i, j) > 0
            pair_list(end+1, :) = [i, j, pair_mass(i, j)]; %#ok<AGROW>
        end
    end
end

if isempty(pair_list)
    return;
end

[~, order] = sort(pair_list(:, 3), 'descend');
pair_list = pair_list(order, :);
pair_list = pair_list(1:min(max_pairs, size(pair_list, 1)), :);

for i = 1:size(pair_list, 1)
    pair_children = children(pair_list(i, 1:2));
    pair_leaf_labels = [];
    for j = 1:numel(pair_children)
        pair_leaf_labels = [pair_leaf_labels; ...
            relift_leaf_descendants(tree, pair_children(j))]; %#ok<AGROW>
    end
    pair_table(end+1).pair_children = pair_children(:)'; %#ok<AGROW>
    pair_table(end).pair_leaf_labels = unique(pair_leaf_labels(:)');
    pair_table(end).conf_mass = pair_list(i, 3) / max(total_weight, eps);
end
end
