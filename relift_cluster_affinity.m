function groups = relift_cluster_affinity(affinity, labels, num_groups)
%RELIFT_CLUSTER_AFFINITY Cluster a local affinity matrix with spectral clustering.

n = size(affinity, 1);
if n == 1
    groups = {labels};
    return;
end

num_groups = min(num_groups, n);
D = diag(sum(affinity, 2));
L = (D - affinity + (D - affinity)') / 2;
[V, eigval] = eig(L);
[~, order] = sort(diag(eigval), 'ascend');
embed = V(:, order(1:num_groups));
cluster_id = kmeans(embed, num_groups, 'Replicates', 20, 'Display', 'off');

groups = cell(1, num_groups);
for i = 1:num_groups
    groups{i} = labels(cluster_id == i)';
end
groups = groups(~cellfun(@isempty, groups));
end
