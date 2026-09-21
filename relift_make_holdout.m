function [fit_data, dev_data, fit_idx, dev_idx] = relift_make_holdout(train_data, val_ratio, seed)
%RELIFT_MAKE_HOLDOUT Deterministic stratified holdout with persisted indices.

if ~isscalar(val_ratio) || ~isfinite(val_ratio) || val_ratio <= 0 || val_ratio >= 1
    error('CoHReHC:InvalidValRatio', 'val_ratio must be strictly between 0 and 1.');
end
if ~isscalar(seed) || ~isfinite(seed) || seed ~= fix(seed)
    error('CoHReHC:InvalidSeed', 'seed must be a finite integer scalar.');
end

rng(seed, 'twister');
labels = unique(train_data(:, end));
fit_idx = [];
dev_idx = [];

for i = 1:numel(labels)
    idx = find(train_data(:, end) == labels(i));
    idx = idx(randperm(numel(idx)));
    dev_num = max(1, round(numel(idx) * val_ratio));
    if numel(idx) - dev_num < 1
        dev_num = max(0, numel(idx) - 1);
    end
    if dev_num < 1
        error('CoHReHC:InsufficientClassSupport', ...
            'Class %.17g cannot supply both fit and dev rows.', labels(i));
    end
    dev_idx = [dev_idx; idx(1:dev_num)]; %#ok<AGROW>
    fit_idx = [fit_idx; idx(dev_num+1:end)]; %#ok<AGROW>
end

fit_data = train_data(fit_idx, :);
dev_data = train_data(dev_idx, :);
end
