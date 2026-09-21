function layer_map = relift_build_layer_map(tree)
%RELIFT_BUILD_LAYER_MAP 按深度把当前树的内部节点分层。
% tree(:,2) 已经是从 root 向下的层级编号，因此这里直接按 depth 分组。
% 返回结果按从深到浅排序，便于 bottom-up 修树。

internal_nodes = tree_InternalNodes(tree);
internal_nodes = [internal_nodes; tree_Root(tree)];
internal_nodes = unique(internal_nodes(:));
depths = tree(internal_nodes, 2);
unique_depths = sort(unique(depths), 'descend');

layer_map = struct('depth', {}, 'node_ids', {}, 'parent_signatures', {}, 'parent_leaf_labels', {});
for i = 1:numel(unique_depths)
    depth = unique_depths(i);
    node_ids = internal_nodes(depths == depth);
    parent_signatures = cell(numel(node_ids), 1);
    parent_leaf_labels = cell(numel(node_ids), 1);
    for j = 1:numel(node_ids)
        leaf_ids = relift_leaf_descendants(tree, node_ids(j));
        parent_leaf_labels{j} = leaf_ids(:)';
        parent_signatures{j} = relift_node_signature(tree, node_ids(j));
    end
    layer_map(end+1).depth = depth; %#ok<AGROW>
    layer_map(end).node_ids = node_ids(:)';
    layer_map(end).parent_signatures = parent_signatures;
    layer_map(end).parent_leaf_labels = parent_leaf_labels;
end
end
