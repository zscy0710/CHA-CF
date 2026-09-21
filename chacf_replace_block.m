function spec = chacf_replace_block(spec, parent, oldChildren, groups)
%CHACF_REPLACE_BLOCK Replace selected children below one parent.

if isnumeric(spec)
    return;
end

if ~isempty(spec.id) && spec.id == parent
    kept = {};
    for i = 1:numel(spec.children)
        child = spec.children{i};
        if ~ismember(local_root(child), oldChildren)
            kept{end+1} = child;
        end
    end
    for i = 1:numel(groups)
        if numel(groups{i}) == 1
            kept{end+1} = groups{i};
        else
            leaves = num2cell(groups{i});
            kept{end+1} = struct('id', [], 'children', {leaves});
        end
    end
    spec.children = kept;
    return;
end

for i = 1:numel(spec.children)
    spec.children{i} = chacf_replace_block( ...
        spec.children{i}, parent, oldChildren, groups);
end
end

function id = local_root(spec)
if isnumeric(spec)
    id = spec;
else
    id = spec.id;
end
end
