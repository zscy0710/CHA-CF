function result = run_awa_mimr()
%RUN_AWA_MIMR Reproduce the AWAphog HFS-MIMR result with CHA-CF.

root = fileparts(mfilename('fullpath'));
train = load(fullfile(root, 'datasets', 'AWAphogTrain.mat'));
test = load(fullfile(root, 'datasets', 'AWAphogTest.mat'));
opt = local_options();

trainData = train.data_array;
testData = test.data_array;
H0 = train.tree;
classCount = tabulate(trainData(:, end));
tailClass = find(classCount(:, 2) <= 0.2 * max(classCount(:, 2)));

fprintf('=== CHA-CF: AWAphog / HFS-MIMR ===\n');
fprintf('rho = %.2f, lambda = %.2f, seed = %d\n', ...
    opt.holdout, opt.fusion, opt.seed);

baseline = local_evaluate(testData, trainData, H0, tailClass, opt);
[fitData, devData] = chacf_holdout(trainData, opt.holdout, opt.seed);
[H1, updates] = chacf_adapt(H0, trainData, fitData, devData, opt);
chacf = local_evaluate(testData, trainData, H1, tailClass, opt);

fprintf('Accepted updates: %d\n', numel(updates));
for i = 1:numel(updates)
    update = updates(i);
    fprintf('  pass %d, depth %d, parent %d, children ', ...
        update.pass, update.depth, update.parent);
    fprintf('%d ', update.children);
    fprintf('| leaves ');
    fprintf('%d ', update.leaves);
    fprintf('| score %.4f\n', update.score);
end

fprintf('\n%-18s %8s %8s %8s %8s\n', 'Method', 'Acc', 'F_H', 'TIE', 'F_LCA');
fprintf('%s\n', repmat('-', 1, 56));
fprintf('%-18s %8.4f %8.4f %8.4f %8.4f\n', ...
    'HFS-MIMR', baseline.acc, baseline.fh, baseline.tie, baseline.flca);
fprintf('%-18s %8.4f %8.4f %8.4f %8.4f\n', ...
    'HFS-MIMR + CHA-CF', chacf.acc, chacf.fh, chacf.tie, chacf.flca);

result = struct();
result.baseline = baseline;
result.chacf = chacf;
result.H0 = H0;
result.H1 = H1;
result.updates = updates;
end

function opt = local_options()
opt.seed = 13;
opt.holdout = 0.25;
opt.fusion = 0.30;
opt.minSupport = 30;
opt.supportFloor = 100;
opt.pairsPerNode = 2;
opt.topCandidates = 8;
opt.topEval = 5;
opt.minScore = 0.12;
opt.overlapThreshold = 0.70;
opt.overlapPenalty = 0.10;
opt.maxLeafOverlap = 0.95;
opt.maxRepeat = 1;
opt.maxPasses = 2;
opt.maxUpdatesPerLevel = 1;
opt.featureCount = 300;
opt.mimrLambda = 10;
opt.mimrAlpha = 0.1;
opt.mimrBeta = 0.01;
opt.mimrIter = 10;
opt.svm = '-c 1 -t 0 -q';
end

function metrics = local_evaluate(testData, trainData, tree, tailClass, opt)
state = chacf_train(trainData, tree, opt);
[acc, ~, flca, fh, tie, ~, ~, ~, ~, ~, ~, ~, ~] = ...
    HierSVMPredictionBatch( ...
    testData, tree, state.feature, tailClass, 'AWAphog', 0);
metrics = struct('acc', acc, 'fh', fh, ...
    'tie', tie / size(testData, 1), 'flca', flca);
end
