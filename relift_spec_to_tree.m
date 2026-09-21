function tree = relift_spec_to_tree(root_spec)
%RELIFT_SPEC_TO_TREE Convert a nested tree back to the matrix representation.

leaf_labels = sort(unique(collect_leaves(root_spec)));
num_leaves = max(leaf_labels);
parent = zeros(4 * num_leaves, 1);
next_id = num_leaves + 1;

root_id = emit_node(root_spec, 0);
parent(root_id) = 0;
parent = parent(1:next_id-1);
tree = [parent, compute_levels(parent)];

    function node_id = emit_node(spec, parent_id)
        if isnumeric(spec)
            node_id = spec;
            parent(node_id) = parent_id;
            return;
        end

        node_id = next_id;
        next_id = next_id + 1;
        if node_id > numel(parent)
            parent = [parent; zeros(num_leaves, 1)]; %#ok<AGROW>
        end

        parent(node_id) = parent_id;
        for i = 1:numel(spec.children)
            emit_node(spec.children{i}, node_id);
        end
    end
end

function leaves = collect_leaves(spec)
if isnumeric(spec)
    leaves = spec;
    return;
end

leaves = [];
for i = 1:numel(spec.children)
    leaves = [leaves, collect_leaves(spec.children{i})]; %#ok<AGROW>
end
end

function levels = compute_levels(parent)
levels = zeros(size(parent));
root = find(parent == 0, 1);
nodes = root;
level = 0;

while ~isempty(nodes)
    next_nodes = find(ismember(parent, nodes));
    level = level + 1;
    levels(next_nodes) = level;
    nodes = next_nodes;
end
end
