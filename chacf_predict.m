function pred = chacf_predict(data, state, startNode)
%CHACF_PREDICT Predict labels below one internal node.

n = size(data, 1);
pred = zeros(n, 1);
leaf = tree_LeafNode(state.tree);

for i = 1:n
    node = startNode;
    while ~ismember(node, leaf)
        selected = state.feature{node}(1:state.nFeature);
        node = svmpredict(0, data(i, selected), state.models{node}, '-q');
    end
    pred(i) = node;
end
end
