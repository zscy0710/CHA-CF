%% Return the index of the middle node.
%% Author: Hong Zhao
%% Date: 2016-5-14
%% Example: 
% load tree;
% return internal nodes and root node
% middleNode = tree_InternalNode( tree )
function [ middleNode ] = tree_InternalNodes( tree )
treeParent = tree(:,1);
root = find(treeParent == 0);

% Internal nodes are exactly the nodes that appear as a parent of some
% other node, excluding the virtual parent 0 and the root node itself.
middleNode = unique(treeParent, 'stable');
middleNode(middleNode == 0) = [];
middleNode(middleNode == root) = [];
end
