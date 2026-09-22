function [accuracyMean, accuracyStd, flcaMean, fhMean, tieMean, testTime, ...
    accuracyLeaf, accuracyTail, fhStd, tieStd, accuracyTailStd, fh, tie] = ...
    chacf_evaluate(data, tree, feature, tailClass, dataset, method)
[n, nFeature] = size(data);
nFeature = nFeature - 1;
if method == 1
    fraction = 1;
elseif strcmp(dataset, 'DD') || strcmp(dataset, 'F194')
    fraction = 0.1;
else
    fraction = 0.2;
end

nSelected = round(nFeature * fraction);
rand('seed', 1);
folds = crossvalind('Kfold', n, 10);
start = tic;
[accuracyMean, accuracyStd, flcaMean, fhMean, tieMean, accuracyLeaf, ...
    accuracyTail, fhStd, tieStd, accuracyTailStd, fh, tie] = ...
    FS_Kflod_TopDownSVMClassifier1( ...
    data, 10, tree, feature, nSelected, folds, tailClass, method);
testTime = toc(start);
end
