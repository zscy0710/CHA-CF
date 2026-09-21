function [fitData, devData] = chacf_holdout(data, ratio, seed)
%CHACF_HOLDOUT Make a stratified development split.

rng(seed, 'twister');
labels = unique(data(:, end));
fitIndex = [];
devIndex = [];

for i = 1:numel(labels)
    index = find(data(:, end) == labels(i));
    index = index(randperm(numel(index)));
    nDev = max(1, round(numel(index) * ratio));
    nDev = min(nDev, numel(index) - 1);
    devIndex = [devIndex; index(1:nDev)];
    fitIndex = [fitIndex; index(nDev+1:end)];
end

fitData = data(fitIndex, :);
devData = data(devIndex, :);
end
