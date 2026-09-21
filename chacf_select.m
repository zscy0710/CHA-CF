function [choice, Hnext, accepted] = chacf_select(tree, trainData, fitData, devData, state, candidates, opt)
%CHACF_SELECT Evaluate the leading candidates on the development split.

baseScore = local_score(tree, state, devData);
bestScore = -Inf;
choice = candidates(1);
Hnext = tree;

for i = 1:min(opt.topEval, numel(candidates))
    trialTree = chacf_rebuild( ...
        tree, trainData, candidates(i), state, devData, fitData, opt.fusion);
    trialState = chacf_train(fitData, trialTree, opt);
    score = local_score(trialTree, trialState, devData);
    if score > bestScore
        bestScore = score;
        choice = candidates(i);
        Hnext = trialTree;
    end
end

accepted = bestScore > baseScore;
if ~accepted
    Hnext = tree;
end
end

function score = local_score(tree, state, devData)
pred = FS_topDownSVMPrediction( ...
    devData(:, 1:end-1), state.models, tree, state.feature, state.nFeature, 0)';
pred = double(pred(:));
[fh, tie] = chacf_local_metrics(tree, tree_Root(tree), devData(:, end), pred);
score = mean(devData(:, end) == pred) + fh - tie;
end
