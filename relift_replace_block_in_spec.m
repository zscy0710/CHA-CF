function spec = relift_replace_block_in_spec(spec, parent_id, block_children, new_groups)
%RELIFT_REPLACE_BLOCK_IN_SPEC 在嵌套树结构里替换一个局部子树块。

if isnumeric(spec)
    return;
end

if ~isempty(spec.id) && spec.id == parent_id
    new_children_specs = {};
    for i = 1:numel(spec.children)
        child_spec = spec.children{i};
        child_id = relift_spec_root_id(child_spec);
        if ismember(child_id, block_children)
            continue;
        end
        new_children_specs{end+1} = child_spec; %#ok<AGROW>
    end
    for i = 1:numel(new_groups)
        group = new_groups{i};
        if numel(group) == 1
            new_children_specs{end+1} = group(1); %#ok<AGROW>
        else
            group_children = cell(1, numel(group));
            for j = 1:numel(group)
                group_children{j} = group(j);
            end
            new_children_specs{end+1} = struct('id', [], 'children', {group_children}); %#ok<AGROW>
        end
    end
    spec.children = new_children_specs;
    return;
end

for i = 1:numel(spec.children)
    spec.children{i} = relift_replace_block_in_spec(spec.children{i}, parent_id, block_children, new_groups);
end
end

function node_id = relift_spec_root_id(spec)
if isnumeric(spec)
    node_id = spec;
else
    node_id = spec.id;
end
end
