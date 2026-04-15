# Phononic Dataset Preprocessing Report

## 处理目标

- 论文样本数：9000
- 输入图像：左上角 256x256，保存为 `(1, 256, 256)` 的 `float32` 二值张量
- 输出标签：1464 点频率序列，单位 kHz，Min-Max 归一化到 `[0, 1]`
- 训练/测试：每类 2900/100，合计 8700/300

## 关键参数

```json
{
  "data_root": "C:\\Users\\fanli\\Desktop\\Dataset_for_surrogate_DL_11000",
  "source": "files",
  "output_dir": "C:\\Users\\fanli\\Desktop\\Dataset_for_surrogate_DL_11000\\processed_phononic_9000",
  "seed": 42,
  "batch_size": 32,
  "image_shape": [
    1,
    256,
    256
  ],
  "label_length": 1464,
  "frequency_key": "F",
  "frequency_scale": 0.001,
  "frequency_unit": "kHz",
  "normalization": "global Min-Max fitted on all 9000 paper samples"
}
```

## 归一化参数

```json
{
  "min_value": 0.0007258703117258847,
  "max_value": 668.60693359375,
  "scale": 668.6062077234383,
  "unit": "kHz"
}
```

## 划分统计

```json
{
  "train_total": 8700,
  "test_total": 300,
  "train_by_group": {
    "cross": 2900,
    "diagonal": 2900,
    "blot": 2900
  },
  "test_by_group": {
    "cross": 100,
    "diagonal": 100,
    "blot": 100
  }
}
```

## 验证结果

```json
{
  "train_images_shape": [
    8700,
    1,
    256,
    256
  ],
  "test_images_shape": [
    300,
    1,
    256,
    256
  ],
  "train_labels_shape": [
    8700,
    1464
  ],
  "test_labels_shape": [
    300,
    1464
  ],
  "image_dtype": "float32",
  "label_dtype": "float32",
  "train_label_min": 0.0,
  "train_label_max": 1.0,
  "test_label_min": 5.156551310392388e-07,
  "test_label_max": 0.9956050515174866,
  "train_image_unique_full_scan": [
    0.0,
    1.0
  ],
  "test_image_unique_full_scan": [
    0.0,
    1.0
  ],
  "train_batch_x_shape": [
    32,
    1,
    256,
    256
  ],
  "train_batch_y_shape": [
    32,
    1464
  ],
  "test_batch_x_shape": [
    32,
    1,
    256,
    256
  ],
  "test_batch_y_shape": [
    32,
    1464
  ],
  "torch_image_dtype": "torch.float32",
  "torch_label_dtype": "torch.float32",
  "cuda_available": true
}
```

## 输出文件

- `train_images.npy`: 2175.00 MB
- `train_labels.npy`: 48.59 MB
- `test_images.npy`: 75.00 MB
- `test_labels.npy`: 1.68 MB
- `frequency_minmax_scaler.npz`: 0.00 MB
- `split_indices.npz`: 0.07 MB
- `split_manifest.csv`: 0.26 MB
- `metadata.json`: 0.00 MB
- `preprocess.log`: 0.00 MB
