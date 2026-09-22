function ok = check_experiment()
%CHECK_EXPERIMENT Compare the experiment output with the reported values.

expected = [ ...
    0.2449, 0.5737, 0.3410; ...
    0.2664, 0.7327, 0.1552];

start = tic;
result = run_experiment();
elapsed = toc(start);
actual = [ ...
    result.baseline.acc, result.baseline.fh, result.baseline.tie; ...
    result.adapted.acc, result.adapted.fh, result.adapted.tie];

ok = all(round(actual, 4) == expected, 'all') && numel(result.updates) == 2;

fprintf('\n%-12s %8s %8s %8s\n', 'method', 'Acc', 'F_H', 'TIE');
fprintf('%s\n', repmat('-', 1, 40));
fprintf('%-12s %8.4f %8.4f %8.4f\n', 'baseline', actual(1, :));
fprintf('%-12s %8.4f %8.4f %8.4f\n', 'CHA-CF', actual(2, :));
fprintf('updates: %d / 2\n', numel(result.updates));
if ok
    fprintf('PASS (%.1f s)\n', elapsed);
else
    fprintf('FAIL (%.1f s)\n', elapsed);
end
end
