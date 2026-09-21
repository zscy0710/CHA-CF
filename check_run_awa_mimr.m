function ok = check_run_awa_mimr()
%CHECK_RUN_AWA_MIMR Compare the AWAphog run with the reported values.

expected = struct( ...
    'baseline_acc', 0.2449, 'baseline_fh', 0.5737, 'baseline_tie', 0.3410, ...
    'chacf_acc', 0.2664, 'chacf_fh', 0.7327, 'chacf_tie', 0.1552, ...
    'blocks', 2);

start_time = tic;
result = run_awa_mimr();
elapsed = toc(start_time);

actual = struct( ...
    'baseline_acc', result.baseline.acc, ...
    'baseline_fh', result.baseline.fh, ...
    'baseline_tie', result.baseline.tie, ...
    'chacf_acc', result.chacf.acc, ...
    'chacf_fh', result.chacf.fh, ...
    'chacf_tie', result.chacf.tie, ...
    'blocks', numel(result.applied_blocks));

names = fieldnames(expected);
ok = true;
fprintf('\n%-14s %-10s %-10s %s\n', 'quantity', 'expected', 'actual', 'verdict');
fprintf('%s\n', repmat('-', 1, 48));
for name_id = 1:numel(names)
    name = names{name_id};
    want = expected.(name);
    got = actual.(name);
    if strcmp(name, 'blocks')
        matched = got == want;
        fprintf('%-14s %-10d %-10d %s\n', name, want, got, local_mark(matched));
    else
        matched = round(got, 4) == want;
        fprintf('%-14s %-10.4f %-10.4f %s\n', name, want, got, local_mark(matched));
    end
    ok = ok && matched;
end

fprintf('%s\n', repmat('-', 1, 48));
if ok
    fprintf('PASS: AWAphog / HFS-MIMR reproduced (%.1f s)\n', elapsed);
else
    fprintf('FAIL: see the rows marked x above (%.1f s)\n', elapsed);
end
end

function mark = local_mark(matched)
if matched
    mark = 'ok';
else
    mark = 'x';
end
end
