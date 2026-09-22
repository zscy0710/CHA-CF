function result = run_experiment()
%RUN_EXPERIMENT Run the CHA-CF experiment on the bundled dataset.

root = fileparts(mfilename('fullpath'));
train = load(fullfile(root, 'datasets', 'F194Train.mat'));
test = load(fullfile(root, 'datasets', 'F194Test.mat'));
opt = options();

trainData = train.data_array;
testData = test.data_array;
H0 = train.tree;
classCount = tabulate(trainData(:, end));
tailClass = find(classCount(:, 2) <= 0.2 * max(classCount(:, 2)));

fprintf('holdout = %.2f, fusion = %.2f, seed = %d\n', ...
    opt.holdout, opt.fusion, opt.seed);

baseline = evaluate(testData, trainData, H0, tailClass, opt);
[fitData, devData] = chacf_holdout(trainData, opt.holdout, opt.seed);
[H1, updates] = chacf_adapt(H0, trainData, fitData, devData, opt);
adapted = evaluate(testData, trainData, H1, tailClass, opt);

fprintf('accepted updates: %d\n', numel(updates));
for i = 1:numel(updates)
    u = updates(i);
    fprintf('  pass %d, depth %d, parent %d, children ', ...
        u.pass, u.depth, u.parent);
    fprintf('%d ', u.children);
    fprintf('| leaves ');
    fprintf('%d ', u.leaves);
    fprintf('| score %.4f\n', u.score);
end

fprintf('\n%-12s %8s %8s %8s %8s\n', 'method', 'Acc', 'F_H', 'TIE', 'F_LCA');
fprintf('%s\n', repmat('-', 1, 50));
fprintf('%-12s %8.4f %8.4f %8.4f %8.4f\n', ...
    'baseline', baseline.acc, baseline.fh, baseline.tie, baseline.flca);
fprintf('%-12s %8.4f %8.4f %8.4f %8.4f\n', ...
    'CHA-CF', adapted.acc, adapted.fh, adapted.tie, adapted.flca);

result = struct();
result.baseline = baseline;
result.adapted = adapted;
result.H0 = H0;
result.H1 = H1;
result.updates = updates;
end

function opt = options()
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
opt.fsLambda = 10;
opt.fsAlpha = 0.1;
opt.fsBeta = 0.01;
opt.fsIter = 10;
opt.svm = '-c 1 -t 0 -q';
end

function metrics = evaluate(testData, trainData, tree, tailClass, opt)
state = chacf_train(trainData, tree, opt);
[acc, ~, flca, fh, tie, ~, ~, ~, ~, ~, ~, ~, ~] = ...
    chacf_evaluate(testData, tree, state.feature, tailClass, 0.2, 0);
metrics = struct('acc', acc, 'fh', fh, ...
    'tie', tie / size(testData, 1), 'flca', flca);
end
