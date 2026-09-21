function [H1, updates] = chacf_adapt(H0, trainData, fitData, devData, opt)
%CHACF_ADAPT Apply bottom-up hierarchy adaptation.

H1 = H0;
updates = struct('pass', {}, 'depth', {}, 'parent', {}, ...
    'children', {}, 'leaves', {}, 'key', {}, 'score', {});

for pass = 1:opt.maxPasses
    changed = false;
    nodes = [tree_InternalNodes(H1); tree_Root(H1)];
    depths = sort(unique(H1(nodes, 2)), 'descend');

    for i = 1:numel(depths)
        depth = depths(i);
        nUpdate = 0;
        while nUpdate < opt.maxUpdatesPerLevel
            state = chacf_train(fitData, H1, opt);
            candidates = chacf_candidates( ...
                H1, state, devData, fitData, updates, depth, opt);
            if isempty(candidates)
                break;
            end

            [choice, Hnext, accepted] = chacf_select( ...
                H1, trainData, fitData, devData, state, candidates, opt);
            if ~accepted
                break;
            end

            H1 = Hnext;
            updates(end+1) = struct( ...
                'pass', pass, 'depth', depth, 'parent', choice.parent, ...
                'children', choice.children, 'leaves', choice.leaves, ...
                'key', choice.key, 'score', choice.score);
            nUpdate = nUpdate + 1;
            changed = true;
        end
    end

    if ~changed
        break;
    end
end
end
