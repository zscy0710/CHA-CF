function [feature_slct, W_full, info] = relift_mimr_feature_select(X, Y, tree, lambda, alpha, beta, maxIte, flag, opts)
%RELIFT_MIMR_FEATURE_SELECT MIMR 的本地可运行封装版本。
% 说明：
% 1. 逻辑保持与 FS/HFS-MIMR-INS-main/HFS-MIMR/MIMR.m 一致。
% 2. 修正原实现里 U2 初始化成 dxd 导致的维度错误。
% 3. 2026-09-07 加入三条实现修复，全部由 opts 控制。
%    **opts.legacy = true 时逐位恢复 2026-09-07 之前的行为**，用于复现既有数字。
%
%    fix1  opts.hoist_gram (默认 true)
%          XX/XY 只依赖 X,Y，是 maxIte 循环的不变量，原实现每轮重算一次。
%          提到循环外。**精确：结果逐位不变。** 同时把 tree_LeafNode(tree)
%          从内层循环（原第 40 行，每轮每节点各调一次）提出，同样精确。
%
%    fix3  opts.skip_empty (默认 true)
%          n_u == 0 的空节点原本照解 d x d 方程，**并且作为兄弟进入真实节点的
%          U1/U2 耦合**（初值 ones(d,maxm) 贡献 maxm*ones(d,d)，之后又被自己的
%          兄弟项驱动出非零 W 反馈回去）。本修复把空节点同时从求解集合与兄弟
%          集合中剔除。空节点自身的 feature_slct 不变（见下方「不变量」）。
%          **会改变真实节点的数字。** 依据：creatSubTablezh -> newlabel_modify_MLNP
%          过滤到 children_set，父节点 SVM 标签集不含空孩子，故空枝在
%          FS_topDownSVMPrediction 中结构上不可达（七集普查已实测确认）。
%
%    fix2  opts.tol (默认 0 = 关闭) / opts.warm_W (默认 {} = 冷启动)
%          收敛即停 + 跨候选热启动。**近似。** 默认关闭，由实测决定是否开启。
%
% 不变量（修复不得破坏，已在 fix_verify_fixes.m 中写成断言）：
%   a) maxm 不受 fix3 影响 —— 空节点 m(u)=numel(unique([]))=0，永不取到最大值；
%   b) 空节点的 feature_slct 不受 fix3 影响 —— 末尾按 m(u)=0 截成 d x 0，
%      sum(...,2) 恒为 zeros(d,1)，sort 稳定 => 恒为 (1:d)'，与原实现一致；
%   c) legacy 模式下兄弟求和的**累加顺序**与原实现完全一致（setdiff 输出升序，
%      剔除元素不改变保留元素的相对次序）—— 浮点加法不满足结合律，顺序必须保住。

persistent fix3_notified   % 每个 MATLAB 会话只提示一次 fix3 生效（见下）

if nargin < 9 || isempty(opts)
    opts = struct();
end
legacy     = local_opt(opts, 'legacy', false);
hoist_gram = local_opt(opts, 'hoist_gram', true) && ~legacy;
skip_empty = local_opt(opts, 'skip_empty', true) && ~legacy;
tol        = local_opt(opts, 'tol', 0);
warm_W     = local_opt(opts, 'warm_W', {});
if legacy
    tol = 0;
    warm_W = {};
end

internalNodes = tree_InternalNodes(tree);
if any(internalNodes == -1)
    % create_SubTable2 第 11 行剔除该哨兵，本函数原先没有；九集数据上从未出现。
    warning('CoHReHC:MimrSentinelNode', ...
        'tree_InternalNodes 返回了 -1 哨兵，已剔除以与 create_SubTable2 对齐。');
    internalNodes(internalNodes == -1) = [];
end
indexRoot = tree_Root(tree);
noLeafNode = [internalNodes; indexRoot];
numSelected = size(X{indexRoot}, 2);
m = zeros(max(noLeafNode), 1);

for i = 1:length(noLeafNode)
    classLabel = unique(Y{noLeafNode(i)});
    m(noLeafNode(i)) = length(classLabel);
end
maxm = max(m);

% fix1: tree_LeafNode 是 tree 的纯函数，原实现在内层循环里每轮每节点各调一次。
leafNode = tree_LeafNode(tree);

% fix3: 空节点普查 + 求解集合 / 兄弟集合的裁剪。
nu = zeros(max(noLeafNode), 1);
for i = 1:length(noLeafNode)
    nu(noLeafNode(i)) = size(X{noLeafNode(i)}, 1);
end
emptyNodes = noLeafNode(nu(noLeafNode) == 0);

if skip_empty
    if nu(indexRoot) == 0
        error('CoHReHC:MimrEmptyRoot', '根节点没有样本，MIMR 无法求解。');
    end
    % fix3 会改变数字，所以每个 MATLAB 会话里第一次真正生效时必须留下痕迹，
    % 不允许静默改变既有结果。
    if ~isempty(emptyNodes) && isempty(fix3_notified)
        fix3_notified = true;
        fprintf(['[relift_mimr] fix3 生效：剔除 %d 个空节点 [%s]（不再解 d x d ' ...
            '方程，也不再进入兄弟耦合 U1/U2）。此项**会改变数字**；' ...
            '设 cfg.mimr_legacy = true 可逐位恢复旧行为。\n'], ...
            numel(emptyNodes), num2str(emptyNodes(:)'));
    end
    solveInternal = setdiff(internalNodes, emptyNodes, 'stable');
    siblingExclude = [leafNode(:); emptyNodes(:)];
else
    solveInternal = internalNodes;
    siblingExclude = leafNode;   % 与原实现第 42 行逐字一致
end
solveNodes = [solveInternal; indexRoot];

warm_hits = 0;
for j = 1:length(noLeafNode)
    [~, d] = size(X{noLeafNode(j)});
    Y{noLeafNode(j)} = conversionY01_extend(Y{noLeafNode(j)}, maxm);
    if numel(warm_W) >= noLeafNode(j) && ~isempty(warm_W{noLeafNode(j)}) && ...
            isequal(size(warm_W{noLeafNode(j)}), [d maxm])
        W{noLeafNode(j)} = warm_W{noLeafNode(j)}; %#ok<AGROW>
        warm_hits = warm_hits + 1;
    else
        W{noLeafNode(j)} = ones(d, maxm); %#ok<AGROW>
    end
end

% fix1: XX/XY 是循环不变量，提到 maxIte 循环外算一次。
if hoist_gram
    for j = 1:length(solveNodes)
        XX{solveNodes(j)} = X{solveNodes(j)}' * X{solveNodes(j)}; %#ok<AGROW>
        XY{solveNodes(j)} = X{solveNodes(j)}' * Y{solveNodes(j)}; %#ok<AGROW>
    end
end

iters_used = maxIte;
deltas = nan(1, maxIte);
for i = 1:maxIte
    if tol > 0
        W_prev = W;
    end

    for j = 1:length(solveNodes)
        D{solveNodes(j)} = diag(0.5 ./ max(sqrt(sum(W{solveNodes(j)} .* W{solveNodes(j)}, 2)), eps)); %#ok<AGROW>
        if ~hoist_gram
            XX{solveNodes(j)} = X{solveNodes(j)}' * X{solveNodes(j)}; %#ok<AGROW>
            XY{solveNodes(j)} = X{solveNodes(j)}' * Y{solveNodes(j)}; %#ok<AGROW>
        end
    end

    d = size(X{indexRoot}, 2);
    W{indexRoot} = (XX{indexRoot} + lambda * D{indexRoot} + beta * ones(d) - beta * eye(d)) \ XY{indexRoot};

    for j = 1:length(solveInternal)
        node_id = solveInternal(j);
        d = size(X{node_id}, 2);
        U1 = zeros(d, d);
        U2 = zeros(d, maxm);
        siblingNodes = tree_Sibling(tree, node_id);
        siblingNodes = setdiff(siblingNodes, siblingExclude);
        for jj = 1:length(siblingNodes)
            sib_id = siblingNodes(jj);
            U1 = U1 + W{sib_id} * W{sib_id}';
            U2 = U2 + W{sib_id};
        end
        W{node_id} = (XX{node_id} + lambda * D{node_id} + beta * (ones(d) - eye(d)) + alpha * U1) \ ...
            (XY{node_id} + alpha * U2);
    end

    if flag == 1
        obj(i) = 1/2 * (norm(X{indexRoot} * W{indexRoot} - Y{indexRoot}))^2 + ...
            lambda * L21(W{indexRoot}) + ...
            beta * trace(ones(d) * W{indexRoot} * W{indexRoot}' - W{indexRoot} * W{indexRoot}'); %#ok<AGROW>
        for j = 1:length(solveInternal)
            currentSibling = tree_Sibling(tree, solveInternal(j));
            currentSibling = setdiff(currentSibling, siblingExclude);
            S = 0;
            for jj = 1:length(currentSibling)
                S = S + norm(W{solveInternal(j)}' * W{currentSibling(jj)} - eye(maxm), 'fro')^2;
            end
            obj(i) = obj(i) + 1/2 * (norm(X{solveInternal(j)} * W{solveInternal(j)} - Y{solveInternal(j)}))^2 + ...
                lambda/2 * L21(W{solveInternal(j)}) + ...
                beta * trace(ones(d) * W{solveInternal(j)} * W{solveInternal(j)}' - W{solveInternal(j)} * W{solveInternal(j)}') + ...
                alpha * S;
        end
    end

    % fix2a: 收敛即停。tol = 0 时 delta < 0 恒假，等价于跑满 maxIte（原行为）。
    if tol > 0
        delta = 0;
        for j = 1:length(solveNodes)
            u = solveNodes(j);
            delta = max(delta, norm(W{u} - W_prev{u}, 'fro') / max(norm(W_prev{u}, 'fro'), eps));
        end
        deltas(i) = delta;
        if delta < tol
            iters_used = i;
            break;
        end
    end
end

% 热启动缓存要的是**截列之前**的完整 W（d x maxm）。
W_full = W;

for i = 1:length(noLeafNode)
    W1 = W{noLeafNode(i)};
    W{noLeafNode(i)} = W1(:, 1:m(noLeafNode(i)));
end

for j = 1:length(noLeafNode)
    tempVector = sum(W{noLeafNode(j)}.^2, 2);
    [~, value] = sort(tempVector, 'descend');
    feature_slct{noLeafNode(j)} = value(1:numSelected); %#ok<AGROW>
end

info = struct();
info.legacy = legacy;
info.hoist_gram = hoist_gram;
info.skip_empty = skip_empty;
info.tol = tol;
info.maxIte = maxIte;
info.iters_used = iters_used;
info.deltas = deltas;
info.maxm = maxm;
info.numSelected = numSelected;
info.num_noleaf = numel(noLeafNode);
info.num_solve_nodes = numel(solveNodes);
info.empty_nodes = emptyNodes(:)';
info.num_empty = numel(emptyNodes);
info.num_empty_skipped = numel(emptyNodes) * double(skip_empty);
info.warm_hits = warm_hits;

if flag == 1 %#ok<UNRCH>
    figure('Color', [1 1 1]);
    plot(obj, 'LineWidth', 4, 'Color', [0 0 1]);
    xlabel('Iteration number');
    ylabel('Objective function value');
end
end

function value = local_opt(options, name, default_value)
value = default_value;
if isstruct(options) && isfield(options, name) && ~isempty(options.(name))
    value = options.(name);
end
end
