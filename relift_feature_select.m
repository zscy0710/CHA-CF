function feature_slct = relift_feature_select(train_data, tree, cfg)
%RELIFT_FEATURE_SELECT Select node-level features with MIMR.

[X, Y] = create_SubTable2(train_data, tree);
feature_slct = relift_mimr_feature_select( ...
    X, Y, tree, cfg.mimr_lambda, cfg.mimr_alpha, cfg.mimr_beta, ...
    cfg.mimr_maxIte, cfg.mimr_flag);
end
