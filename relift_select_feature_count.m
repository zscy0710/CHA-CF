function numberSel = relift_select_feature_count(numFeature, dataset_name, fs_cfg)
%RELIFT_SELECT_FEATURE_COUNT 按原始 HCP 规则确定每个节点选多少特征。

if nargin >= 3 && isfield(fs_cfg, 'feature_select_ratio') && ~isempty(fs_cfg.feature_select_ratio)
    numberSel = round(numFeature * fs_cfg.feature_select_ratio);
    numberSel = max(numberSel, 1);
    return;
end

if strcmp(dataset_name, 'DD') || strcmp(dataset_name, 'F194')
    ratio_flag = 1;
else
    ratio_flag = 2;
end

numberSel = round(numFeature * ratio_flag * 0.1);
numberSel = max(numberSel, 1);
end
