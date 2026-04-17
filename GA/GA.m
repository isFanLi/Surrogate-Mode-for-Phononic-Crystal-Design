clear; clc;

disp(' 正在读取 PyTorch 的 ONNX 模型...');
% 导入模型 
net = importNetworkFromONNX('phononic_cnn.onnx');

nvars = 256; % 基因数量 (16x16)

% 限制基因的突变范围在 0 到 1 之间
lb = zeros(1, nvars);
ub = ones(1, nvars);

% 遗传算法超参数配置
options = optimoptions('ga', ...
    'PopulationSize', 400, ...          % 种群大小：400
    'MaxGenerations', 300, ...          % 繁衍代数：300代 (先少跑一点试试水)
    'Display', 'iter', ...             % 在命令行实时打印每一代的成绩
    'PlotFcn', @gaplotbestf,...
    'UseParallel', true);          % 实时弹窗画出“分数进化图”

% 打分器匿名函数
FitnessFcn = @(dna) calculate_bandgap(dna, net);

disp('开始进行拓扑优化...');
[best_dna, best_score] = ga(FitnessFcn, nvars, [], [], [], [], lb, ub, [], [], options);

% ==========================================
% 结果展示
% ==========================================
final_bandgap = -best_score;
fprintf('找到的最大带隙为: %f\n', final_bandgap);

% 画出进化出的最终物理结构
best_matrix = reshape(best_dna, [16, 16]);
best_img_256 = imresize(best_matrix, [256, 256], 'bilinear');
binary_best_img = best_img_256 > 0.5;

figure;
imshow(binary_best_img);
title(sprintf('Optimized Phononic Crystal\nBandgap: %.4f', final_bandgap));