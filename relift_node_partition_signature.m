function signature = relift_node_partition_signature(tree, node_id)
%RELIFT_NODE_PARTITION_SIGNATURE 内部节点的**子问题**稳定签名（跨候选热启动用）。
%
% 与 relift_node_signature 的区别：后者只签「本节点覆盖哪些叶标签」，
% 而 MIMR 在节点 u 上解的子问题由 (X_u, Y_u) 决定 —— Y_u 是**孩子划分**，
% 所以叶集合相同、划分不同的两个节点是**不同的**子问题，不能互相热启动。
%
% 签名 = 按孩子编号升序排列的各孩子叶集合。之所以按升序：
% conversionY01_extend 用 unique(Y_u) 决定 one-hot 的列序，unique 升序排列，
% 而 Y_u 的取值就是孩子编号 —— 故「升序孩子」与「列序」一一对应。
% 于是**签名相同 ⟹ 子问题逐位相同（含列语义）**，热启动只是换了个初值，
% 不改变不动点。签名不同则回退冷启动（保守，绝不会错配）。
%
% 修树后内部节点编号会被 relift_spec_to_tree 重排，故不得用 node_id 作键。

children = sort(get_children_set(tree, node_id));
parts = strings(1, numel(children));
for i = 1:numel(children)
    leaf_ids = sort(unique(relift_leaf_descendants(tree, children(i))));
    parts(i) = strjoin(string(leaf_ids(:)'), '_');
end
signature = char(strjoin(parts, '|'));
end
