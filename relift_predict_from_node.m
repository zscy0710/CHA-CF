function pred = relift_predict_from_node(eval_x, state, start_node)
%RELIFT_PREDICT_FROM_NODE Predict downward from one internal tree node.

sample_count = size(eval_x, 1);
pred = zeros(sample_count, 1);
tree = state.tree;
leaf_nodes = tree_LeafNode(tree);

for sample_id = 1:sample_count
    current_node = start_node;
    while ~ismember(current_node, leaf_nodes)
        selected = state.feature_slct{current_node}( ...
            1:min(state.numberSel, numel(state.feature_slct{current_node})));
        current_node = svmpredict( ...
            0, eval_x(sample_id, selected), state.models{current_node}, '-q');
    end
    pred(sample_id) = current_node;
end
end
