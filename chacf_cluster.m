function groups = chacf_cluster(affinity, labels, nGroup)
%CHACF_CLUSTER Regroup a local affinity matrix by spectral clustering.

n = size(affinity, 1);
if n == 1
    groups = {labels};
    return;
end

nGroup = min(nGroup, n);
L = diag(sum(affinity, 2)) - affinity;
L = (L + L') / 2;
[V, D] = eig(L);
[~, order] = sort(diag(D), 'ascend');
embedding = V(:, order(1:nGroup));
cluster = kmeans(embedding, nGroup, 'Replicates', 20, 'Display', 'off');

groups = cell(1, nGroup);
for i = 1:nGroup
    groups{i} = labels(cluster == i)';
end
groups = groups(~cellfun(@isempty, groups));
end
