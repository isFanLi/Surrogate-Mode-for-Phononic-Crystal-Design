function negative_bandgap = calculate_bandgap(dna_vector, net)
    % 1. 基因重组：把 1D 的 256 个基因变成 16x16 矩阵
    dna_matrix = reshape(dna_vector, [16, 16]);
    
    % 2. 物理放大：放大到 256x256
    img_256 = imresize(dna_matrix, [256, 256], 'bilinear');
    
    % 3. 二值化：大于0.5变材料(1)，小于0.5变空气(0)
    binary_img = single(img_256 > 0.5); 
    

    formatted_input = reshape(binary_img, [256, 256, 1, 1]);
    
    % 4. 召唤替身预测
    preds = predict(net, formatted_input); 
    
    % 5. 脱壳：新版 MATLAB 预测出的可能是 dlarray 类型，需要脱壳成普通数组
    if isdlarray(preds)
        preds = extractdata(preds);
    end
    
    % 6. 物理切片：把 1464 切成 61 个 K 点 x 24 条能带
    preds_vector = preds(:); % 强行拉平，防止行列颠倒报错
    preds_matrix = reshape(preds_vector, [61, 24])'; % 转置为 24x61
    
    % 7. 计算目标：第4条的最高点，和第5条的最低点
    band4_max = max(preds_matrix(4, :));
    band5_min = min(preds_matrix(5, :));
    
    bandgap = band5_min - band4_max;
    
    % 返回负的带隙值供 GA 寻优
    negative_bandgap = -bandgap; 
end