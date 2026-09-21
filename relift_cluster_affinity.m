function groups = relift_cluster_affinity(affinity, labels, num_groups)
%RELIFT_CLUSTER_AFFINITY 对一个局部块做谱聚类分组。
% 先由相似度矩阵构造拉普拉斯嵌入，再用 k-means 得到新的叶子分组。

n = size(affinity, 1);
if n == 1
    groups = {labels};
    return;
end

num_groups = min(num_groups, n);
D = diag(sum(affinity, 2));
L = D - affinity;
L = (L + L') / 2;
[V, eigval] = eig(L);
[~, order] = sort(diag(eigval), 'ascend');
embed = V(:, order(1:num_groups));
cluster_id = kmeans(embed, num_groups, 'Replicates', 20, 'Display', 'off');

% 把聚类编号再还原成原始叶标签的分组。
groups = cell(1, num_groups);
for i = 1:num_groups
    groups{i} = labels(cluster_id == i)';
end
groups = groups(~cellfun(@isempty, groups));
end
