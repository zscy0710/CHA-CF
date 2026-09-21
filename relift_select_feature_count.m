function numberSel = relift_select_feature_count(num_feature, dataset_name, cfg)
%RELIFT_SELECT_FEATURE_COUNT Choose the number of features used at each node.

if ~isempty(cfg.feature_select_ratio)
    numberSel = max(1, round(num_feature * cfg.feature_select_ratio));
    return;
end

if strcmpi(dataset_name, 'AWAphog')
    numberSel = min(300, num_feature);
else
    numberSel = min(200, num_feature);
end
end
