%% creatSubTable
% Written by Yu Wang
% Modified by Hong Zhao
% Modified by Haoyang Liu
%% Creat subtable
function [DataMod,LabelMod,IndxMod]=create_SubTable2(dataset, tree)
Data = dataset(:,1:end-1);
Label =  dataset(:,end);
[numTrain,~] = size(dataset);
internalNodes = tree_InternalNodes(tree);
internalNodes(find(internalNodes==-1))=[];
indexRoot = tree_Root(tree);% The root of the tree
noLeafNode =[internalNodes;indexRoot];
leafnodes=tree_LeafNode( tree );
for i = 1:length(leafnodes)
    cur_descendants = leafnodes(i);
    ind_d = 1;  % index for id subscript increment
    id = [];        % data whose labels belong to the descendants of the current nodes
    for n = 1:numTrain
        if (ismember(Label(n), cur_descendants) ~= 0)
            id(ind_d) =  n;
            ind_d = ind_d +1;
        end
    end
    DataMod{leafnodes(i)} = Data(id, :);   % Find the sub training set of relative to-be-classified nodes
    LabelMod{leafnodes(i)} = Label(id, :);
    IndxMod{leafnodes(i)} = id;
end
for i = 1:length(noLeafNode)
    cur_descendants = tree_Descendant(tree, noLeafNode(i));
    ind_d = 1;  % index for id subscript increment
    id = [];        % data whose labels belong to the descendants of the current nodes
    for n = 1:numTrain
        if (ismember(Label(n), cur_descendants) ~= 0)
            id(ind_d) =  n;
            ind_d = ind_d +1;
        end
    end
    Label_Uni_Sel = Label(id,:);
    DataSel = Data(id,:);     %select relative training data for the current classifier
    numTrainSel = size(Label_Uni_Sel,1);
    LabelUniSelMod = mylabel_modify_MLNP(Label_Uni_Sel, noLeafNode(i), tree);%% 
    % Get the sub-training set containing only relative nodes
    ind_tdm = 1;
    index = [];     % data whose labels belong to the children of the current nodes
    children_set = get_children_set(tree, noLeafNode(i));
    for ns = 1:numTrainSel
        if (ismember(LabelUniSelMod(ns), children_set) ~= 0)
            index(ind_tdm) =  ns;
            ind_tdm = ind_tdm +1;
        end
    end
    DataMod{noLeafNode(i)} = DataSel(index, :);   % Find the sub training set of relative to-be-classified nodes
    LabelMod{noLeafNode(i)} = LabelUniSelMod(index, :);
    IndxMod{noLeafNode(i)} = index;
end
end