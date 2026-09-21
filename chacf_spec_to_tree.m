function tree = chacf_spec_to_tree(spec)
%CHACF_SPEC_TO_TREE Convert a nested tree to the matrix representation.

leaves = sort(unique(local_leaves(spec)));
nLeaf = max(leaves);
parent = zeros(4 * nLeaf, 1);
next = nLeaf + 1;
root = local_emit(spec, 0);
parent(root) = 0;
parent = parent(1:next-1);
tree = [parent, local_level(parent)];

    function node = local_emit(part, parentNode)
        if isnumeric(part)
            node = part;
            parent(node) = parentNode;
            return;
        end

        node = next;
        next = next + 1;
        if node > numel(parent)
            parent = [parent; zeros(nLeaf, 1)];
        end
        parent(node) = parentNode;
        for k = 1:numel(part.children)
            local_emit(part.children{k}, node);
        end
    end
end

function leaves = local_leaves(spec)
if isnumeric(spec)
    leaves = spec;
    return;
end
leaves = [];
for i = 1:numel(spec.children)
    leaves = [leaves, local_leaves(spec.children{i})];
end
end

function level = local_level(parent)
level = zeros(size(parent));
nodes = find(parent == 0, 1);
depth = 0;
while ~isempty(nodes)
    nodes = find(ismember(parent, nodes));
    depth = depth + 1;
    level(nodes) = depth;
end
end
