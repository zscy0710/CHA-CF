function affinity = chacf_feature_affinity(data, labels)
%CHACF_FEATURE_AFFINITY Compute feature-space affinity for a candidate block.

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
