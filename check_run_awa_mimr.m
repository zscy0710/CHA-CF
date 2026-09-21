function ok = check_run_awa_mimr()
%CHECK_RUN_AWA_MIMR Compare the AWAphog run with the reported values.

expected = [ ...
    0.2449, 0.5737, 0.3410; ...
    0.2664, 0.7327, 0.1552];

start = tic;
result = run_awa_mimr();
elapsed = toc(start);
actual = [ ...
    result.baseline.acc, result.baseline.fh, result.baseline.tie; ...
    result.chacf.acc, result.chacf.fh, result.chacf.tie];

names = {'HFS-MIMR', 'HFS-MIMR + CHA-CF'};
metrics = {'Acc', 'F_H', 'TIE'};
ok = all(round(actual, 4) == expected, 'all') && numel(result.updates) == 2;

fprintf('\n%-20s', 'Method');
fprintf('%10s', metrics{:});
fprintf('\n%s\n', repmat('-', 1, 50));
for i = 1:size(actual, 1)
    fprintf('%-20s', names{i});
    fprintf('%10.4f', actual(i, :));
    fprintf('\n');
end
fprintf('updates: %d / 2\n', numel(result.updates));
if ok
    fprintf('PASS (%.1f s)\n', elapsed);
else
    fprintf('FAIL (%.1f s)\n', elapsed);
end
end
