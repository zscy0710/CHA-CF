function ok = check_run_awa_mimr()
%CHECK_RUN_AWA_MIMR Run the retained setting and verify its paper results.
% Add the project root to the MATLAB path, then call:
%   check_run_awa_mimr();
%
% The check compares the six published metric values and the number of
% accepted blocks, at the four decimal places reported in the paper.

expected = struct( ...
    'baseline_acc', 0.2449, 'baseline_fh', 0.5737, 'baseline_tie', 0.3410, ...
    'chacf_acc', 0.2664, 'chacf_fh', 0.7327, 'chacf_tie', 0.1552, ...
    'blocks', 2);

timer_id = tic;
result = run_awa_mimr();
elapsed = toc(timer_id);

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
fprintf('\n%-14s %-10s %-10s %s\n', 'quantity', 'paper', 'actual', 'verdict');
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

% In this release H0 is the dataset hierarchy, so this duplicate evaluation
% must agree with the original-tree baseline.
h0_is_original = round(result.initial.acc, 4) == round(result.baseline.acc, 4) && ...
    round(result.initial.fh, 4) == round(result.baseline.fh, 4) && ...
    round(result.initial.tie, 4) == round(result.baseline.tie, 4);
fprintf('%-14s %-10s %-10s %s\n', 'H0==baseline', 'yes', ...
    local_yesno(h0_is_original), local_mark(h0_is_original));
ok = ok && h0_is_original;

fprintf('%s\n', repmat('-', 1, 48));
if ok
    fprintf('PASS: reproduces the published AWAphog x HFS-MIMR result (%.1f s)\n', elapsed);
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

function text = local_yesno(value)
if value
    text = 'yes';
else
    text = 'no';
end
end
