function layer_map = relift_build_layer_map(tree)
%RELIFT_BUILD_LAYER_MAP Group internal nodes by depth for bottom-up adaptation.

internal_nodes = [tree_InternalNodes(tree); tree_Root(tree)];
internal_nodes = unique(internal_nodes(:));
depths = tree(internal_nodes, 2);
unique_depths = sort(unique(depths), 'descend');

layer_map = struct('depth', {}, 'node_ids', {});
for i = 1:numel(unique_depths)
    depth = unique_depths(i);
    layer_map(end+1).depth = depth; %#ok<AGROW>
    layer_map(end).node_ids = internal_nodes(depths == depth)';
end
end
