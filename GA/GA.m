clear; clc;

disp(' 正在读取 PyTorch 的 ONNX 模型...');
% 导入模型 (注意：这可能需要安装 MATLAB 的 ONNX Converter Add-on)
net = importNetworkFromONNX('..\phononic_cnn.onnx');
disp('✅ 模型导入成功！遗传算法即将启动...');

% ==========================================
% 设置遗传算法参数
% ==========================================
nvars = 256; % 基因数量 (16x16)

% 限制基因的突变范围在 0 到 1 之间
lb = zeros(1, nvars);
ub = ones(1, nvars);

% 遗传算法超参数配置
options = optimoptions('ga', ...
    'PopulationSize', 400, ...          % 种群大小：400
    'MaxGenerations', 300, ...          % 繁衍代数：300代 (先少跑一点试试水)
    'Display', 'iter', ...             % 在命令行实时打印每一代的成绩
    'PlotFcn', @gaplotbestf);          % 实时弹窗画出“分数进化图”

% ==========================================
% 启动创世纪大循环！
% ==========================================
% 因为我们的打分器需要 net 这个额外参数，所以用 @(x) 把它打包成匿名函数
FitnessFcn = @(dna) calculate_bandgap(dna, net);

disp('🔥 开始进行拓扑优化...');
[best_dna, best_score] = ga(FitnessFcn, nvars, [], [], [], [], lb, ub, [], [], options);

% ==========================================
% 结果展示
% ==========================================
final_bandgap = -best_score;
fprintf('🎉 优化彻底完成！找到的最大带隙为: %f\n', final_bandgap);

% 画出进化出的最终物理结构
best_matrix = reshape(best_dna, [16, 16]);
best_img_256 = imresize(best_matrix, [256, 256], 'bilinear');
binary_best_img = best_img_256 > 0.5;

figure;
imshow(binary_best_img);
title(sprintf('Optimized Phononic Crystal\nBandgap: %.4f', final_bandgap));