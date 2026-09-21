function signature = relift_node_signature(tree, node_id)
%RELIFT_NODE_SIGNATURE 用叶标签集合为当前子树生成稳定签名。
% 修树后内部节点编号会重排，所以跨轮不能依赖 node_id。
% 这里把该节点覆盖的叶标签集合转成字符串，作为稳定标识。

leaf_ids = relift_leaf_descendants(tree, node_id);
leaf_ids = sort(unique(leaf_ids(:)'));
parts = strings(1, numel(leaf_ids));
for i = 1:numel(leaf_ids)
    parts(i) = string(leaf_ids(i));
end
signature = char(strjoin(parts, '_'));
end
