
function [label_cell] = label_class(data)
% 找出所有不同的标签类别
labels = unique(data(:, end));

% 创建一个cell数组，用于存储不同标签对应的样本集合
label_cell = cell(length(labels), 1);

% 将数据按标签分类存储
for i = 1:size(data, 1)
    label = data(i, end);
    label_idx = find(labels == label);
    if isempty(label_cell{label_idx})
        label_cell{label_idx} = data(i, :);
    else
        label_cell{label_idx} = [label_cell{label_idx}; data(i, :)];
    end
end
end