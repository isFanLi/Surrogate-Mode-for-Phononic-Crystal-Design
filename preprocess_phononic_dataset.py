"""
声子晶体拓扑优化数据集预处理脚本。

目标：
1. 只使用论文中采用的 9000 个样本：
   - cross-like    : case 1001-4000，共 3000 个
   - diagonal-like : case 4001-7000，共 3000 个
   - blot-like     : case 8001-11000，共 3000 个
2. 输入 X：
   - 原图为 512x512 二值图像时，截取左上角四分之一，得到 256x256。
   - 转成 PyTorch 需要的 CHW 格式，最终形状为 (1, 256, 256)。
   - 保存为 float32，像素严格为 0.0 或 1.0。
3. 输出 Y：
   - 每个样本读取 1464 点频率向量。
   - 将 .mat 中的 F 从 Hz 转成 kHz，使最大值约为 668.6 kHz。
   - 用全体 9000 个样本的频率做 Min-Max 归一化，保证保存后的标签落在 [0, 1]。
   - 保存 Min-Max 参数和反归一化逻辑，便于后续恢复真实频率。
4. 划分：
   - 每一类随机抽取 2900 个训练样本，剩余 100 个测试样本。
   - 最终训练集 8700，测试集 300。
5. 留档：
   - 保存 split_manifest.csv、metadata.json、preprocess_report.md、preprocess.log。

说明：
论文 Prompt 中可以假设原始数据已整理为两个大文件：
    images = np.load("images.npy", mmap_mode="r")
    labels = np.load("labels.npy")

本脚本保留了该 .npy 加载入口；但当前目录真实数据是 inputs/*.png 和
outputs/*.mat，所以默认 source=files。
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Literal

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# -----------------------------
# 1. 固定配置：严格对应论文数据范围和任务形状
# -----------------------------

IMAGE_SIZE = 256
LABEL_LENGTH = 1464
BATCH_SIZE = 32

# README 中论文使用的 9000 个 case 范围；每类正好 3000 个。
CASE_GROUPS = {
    "cross": range(1001, 4001),
    "diagonal": range(4001, 7001),
    "blot": range(8001, 11001),
}

# 当前 .mat 文件中的 F 数值最大约 668607，单位按 Hz 处理；
# 乘以 0.001 后保存为 kHz，最大值约 668.607 kHz。
MAT_FREQUENCY_SCALE_TO_KHZ = 1.0e-3


@dataclass(frozen=True)
class SampleRecord:
    """记录一个论文样本的 case 编号、类别和原始位置。"""

    source_index: int
    case_id: int
    group: str
    group_id: int


@dataclass(frozen=True)
class FrequencyMinMaxScaler:
    """频率 Min-Max 归一化器，并保留反归一化逻辑。"""

    min_value: float
    max_value: float
    unit: str = "kHz"

    @property
    def scale(self) -> float:
        # 防止极端情况下 max == min 导致除零。
        return max(self.max_value - self.min_value, 1e-12)

    @classmethod
    def fit(cls, values: np.ndarray, unit: str = "kHz") -> "FrequencyMinMaxScaler":
        values = np.asarray(values, dtype=np.float32)
        return cls(
            min_value=float(values.min()),
            max_value=float(values.max()),
            unit=unit,
        )

    def transform(self, values: np.ndarray) -> np.ndarray:
        """将真实频率值缩放到 [0, 1]。"""
        values = np.asarray(values, dtype=np.float32)
        return ((values - self.min_value) / self.scale).astype(np.float32)

    def inverse_transform(
        self, values: np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        """将归一化频率恢复到真实频率，单位见 self.unit。"""
        if torch.is_tensor(values):
            return values * self.scale + self.min_value
        values = np.asarray(values, dtype=np.float32)
        return (values * self.scale + self.min_value).astype(np.float32)

    def to_dict(self) -> dict[str, float | str]:
        return {
            "min_value": self.min_value,
            "max_value": self.max_value,
            "scale": self.scale,
            "unit": self.unit,
        }

    def save(self, path: Path) -> None:
        """保存归一化参数，训练和推理阶段都要使用同一份 scaler。"""
        np.savez(
            path,
            min_value=np.array(self.min_value, dtype=np.float32),
            max_value=np.array(self.max_value, dtype=np.float32),
            unit=np.array(self.unit),
        )

    @classmethod
    def load(cls, path: Path) -> "FrequencyMinMaxScaler":
        data = np.load(path)
        return cls(
            min_value=float(data["min_value"]),
            max_value=float(data["max_value"]),
            unit=str(data["unit"]),
        )


class PhononicDataset(Dataset):
    """
    预处理后数据的 PyTorch Dataset。

    __getitem__ 返回：
    - image: torch.float32, shape=(1, 256, 256), values in {0, 1}
    - label: torch.float32, shape=(1464,), values in [0, 1]
    """

    def __init__(self, images_path: Path, labels_path: Path) -> None:
        # mmap_mode="r" 避免一次性把 2GB 以上图像全部读入内存。
        self.images = np.load(images_path, mmap_mode="r")
        self.labels = np.load(labels_path, mmap_mode="r")

        if self.images.shape[0] != self.labels.shape[0]:
            raise ValueError(
                f"Image count {self.images.shape[0]} != label count {self.labels.shape[0]}"
            )

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # np.load(..., mmap_mode="r") 返回只读视图；copy 后再交给 torch，
        # 避免 PyTorch 对 non-writable NumPy array 的警告和潜在副作用。
        image_np = np.asarray(self.images[index], dtype=np.float32).copy()
        label_np = np.asarray(self.labels[index], dtype=np.float32).copy()
        image = torch.from_numpy(image_np)
        label = torch.from_numpy(label_np)
        return image, label


def build_dataloaders(
    processed_dir: Path,
    batch_size: int = BATCH_SIZE,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """用预处理后的 .npy 文件构建训练和测试 DataLoader。"""
    train_dataset = PhononicDataset(
        processed_dir / "train_images.npy",
        processed_dir / "train_labels.npy",
    )
    test_dataset = PhononicDataset(
        processed_dir / "test_images.npy",
        processed_dir / "test_labels.npy",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader


def setup_logging(output_dir: Path) -> None:
    """同时把运行过程输出到终端和 preprocess.log，作为操作留档。"""
    log_path = output_dir / "preprocess.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def build_paper_records() -> list[SampleRecord]:
    """构造论文使用的 9000 个样本清单。"""
    records: list[SampleRecord] = []
    source_index = 0
    for group_id, (group, case_range) in enumerate(CASE_GROUPS.items()):
        for case_id in case_range:
            records.append(
                SampleRecord(
                    source_index=source_index,
                    case_id=int(case_id),
                    group=group,
                    group_id=group_id,
                )
            )
            source_index += 1

    if len(records) != 9000:
        raise RuntimeError(f"Expected 9000 paper samples, got {len(records)}")
    return records


def split_records(
    records: list[SampleRecord],
    seed: int,
    train_per_group: int = 2900,
    test_per_group: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """按类别分层抽样：每类 2900 训练，100 测试。"""
    rng = np.random.default_rng(seed)
    group_names = list(CASE_GROUPS.keys())
    train_rows: list[np.ndarray] = []
    test_rows: list[np.ndarray] = []

    groups = np.asarray([record.group for record in records])
    for group in group_names:
        rows = np.flatnonzero(groups == group)
        expected = train_per_group + test_per_group
        if rows.size != expected:
            raise RuntimeError(f"{group} expected {expected} samples, got {rows.size}")

        shuffled = rows.copy()
        rng.shuffle(shuffled)
        train_rows.append(shuffled[:train_per_group])
        test_rows.append(shuffled[train_per_group:])

    train_indices = np.concatenate(train_rows).astype(np.int64)
    test_indices = np.concatenate(test_rows).astype(np.int64)
    return train_indices, test_indices


def infer_npy_source_indices(num_samples: int) -> np.ndarray:
    """
    对 images.npy / labels.npy 的样本顺序做最小假设：
    - 若 N=9000，认为它已经只包含论文子集，顺序为 cross/diagonal/blot。
    - 若 N=11000，认为它按 case_id 1..11000 排列，从中抽取论文 case。
    """
    paper_case_ids = np.asarray([record.case_id for record in build_paper_records()])
    if num_samples == 9000:
        return np.arange(9000, dtype=np.int64)
    if num_samples == 11000:
        return paper_case_ids - 1
    raise ValueError(f"Expected 9000 or 11000 samples in .npy source, got {num_samples}")


def load_packed_npy_source(
    data_root: Path,
    images_name: str = "images.npy",
    labels_name: str = "labels.npy",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    加载 Prompt 中假设的两个大 .npy 文件。

    注意：images 使用 mmap，只在写入预处理结果时按样本读取；
    labels 只有约 50MB，可以直接读入内存，便于拟合 Min-Max scaler。
    """
    images_path = data_root / images_name
    labels_path = data_root / labels_name
    images = np.load(images_path, mmap_mode="r")
    labels = np.asarray(np.load(labels_path), dtype=np.float32).reshape(images.shape[0], -1)

    if labels.shape[1] != LABEL_LENGTH:
        raise ValueError(f"Expected labels length {LABEL_LENGTH}, got {labels.shape}")

    source_indices = infer_npy_source_indices(images.shape[0])
    return images, labels[source_indices], source_indices


def load_png_image(path: Path) -> np.ndarray:
    """读取单张 PNG，并转为灰度数组。"""
    with Image.open(path) as image:
        return np.asarray(image.convert("L"))


def preprocess_image_to_chw(image: np.ndarray) -> np.ndarray:
    """截取左上角 256x256，并转为 float32 的 (1, 256, 256) 二值图。"""
    image = np.asarray(image)

    # 兼容 HWC 或 CHW，但当前原始 PNG 实际读出来是 2D 灰度图。
    if image.ndim == 3:
        if image.shape[0] in (1, 3, 4):
            image = image[0]
        elif image.shape[-1] in (1, 3, 4):
            image = image[..., 0]
        else:
            raise ValueError(f"Cannot infer image channel dimension from {image.shape}")

    if image.ndim != 2:
        raise ValueError(f"Expected 2D image after grayscale conversion, got {image.shape}")
    if image.shape[0] < IMAGE_SIZE or image.shape[1] < IMAGE_SIZE:
        raise ValueError(f"Image too small for 256x256 crop: {image.shape}")

    cropped = image[:IMAGE_SIZE, :IMAGE_SIZE]
    binary = (cropped > 0).astype(np.float32)
    return binary[None, :, :]


def load_frequency_from_mat(
    mat_path: Path,
    frequency_key: str = "F",
    frequency_scale: float = MAT_FREQUENCY_SCALE_TO_KHZ,
) -> np.ndarray:
    """读取 .mat 中的 1464 点频率向量，并按比例转换单位。"""
    try:
        with h5py.File(mat_path, "r") as handle:
            values = np.asarray(handle[frequency_key]).reshape(-1)
    except OSError:
        # 少数非 MATLAB 7.3 文件可用 scipy.io.loadmat 读取。
        from scipy.io import loadmat

        values = np.asarray(loadmat(mat_path)[frequency_key]).reshape(-1)

    values = values.astype(np.float32) * np.float32(frequency_scale)
    if values.shape != (LABEL_LENGTH,):
        raise ValueError(f"{mat_path} expected label shape (1464,), got {values.shape}")
    return values


def load_file_labels(
    data_root: Path,
    records: list[SampleRecord],
    frequency_key: str,
    frequency_scale: float,
) -> np.ndarray:
    """先读取全部 9000 个频率向量，用于拟合全局 Min-Max scaler。"""
    output_dir = data_root / "outputs"
    labels = np.empty((len(records), LABEL_LENGTH), dtype=np.float32)

    for row, record in enumerate(tqdm(records, desc="读取频率标签", unit="sample")):
        mat_path = output_dir / f"out_lines_{record.case_id}_a8_h3.mat"
        labels[row] = load_frequency_from_mat(
            mat_path,
            frequency_key=frequency_key,
            frequency_scale=frequency_scale,
        )
    return labels


def get_raw_image(
    source: Literal["files", "npy"],
    data_root: Path,
    records: list[SampleRecord],
    source_images: np.ndarray | None,
    source_indices: np.ndarray | None,
    row: int,
) -> np.ndarray:
    """根据 source 类型读取原始图像。"""
    if source == "files":
        image_path = data_root / "inputs" / f"PC_label_{records[row].case_id}.png"
        return load_png_image(image_path)

    if source_images is None or source_indices is None:
        raise ValueError("source_images and source_indices are required for source='npy'")
    return np.asarray(source_images[int(source_indices[row])])


def write_split_manifest(
    path: Path,
    records: list[SampleRecord],
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> None:
    """保存每个样本属于哪个 split，方便后续复现实验划分。"""
    split_rows = [("train", train_indices), ("test", test_indices)]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "array_index", "source_index", "case_id", "group", "group_id"])
        for split_name, rows in split_rows:
            for array_index, source_index in enumerate(rows):
                record = records[int(source_index)]
                writer.writerow(
                    [
                        split_name,
                        array_index,
                        record.source_index,
                        record.case_id,
                        record.group,
                        record.group_id,
                    ]
                )


def write_processed_arrays(
    *,
    source: Literal["files", "npy"],
    data_root: Path,
    output_dir: Path,
    records: list[SampleRecord],
    labels_khz: np.ndarray,
    scaler: FrequencyMinMaxScaler,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    source_images: np.ndarray | None,
    source_indices: np.ndarray | None,
) -> None:
    """把处理后的 train/test 图像和标签写成 .npy。"""
    split_specs = {
        "train": train_indices,
        "test": test_indices,
    }

    for split_name, rows in split_specs.items():
        image_path = output_dir / f"{split_name}_images.npy"
        label_path = output_dir / f"{split_name}_labels.npy"

        # open_memmap 会直接创建标准 .npy 文件，避免大数组一次性占用内存。
        image_array = np.lib.format.open_memmap(
            image_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(rows), 1, IMAGE_SIZE, IMAGE_SIZE),
        )
        label_array = np.lib.format.open_memmap(
            label_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(rows), LABEL_LENGTH),
        )

        for out_row, source_row in enumerate(
            tqdm(rows, desc=f"写入 {split_name} 数据", unit="sample")
        ):
            source_row = int(source_row)
            raw_image = get_raw_image(
                source=source,
                data_root=data_root,
                records=records,
                source_images=source_images,
                source_indices=source_indices,
                row=source_row,
            )
            image_array[out_row] = preprocess_image_to_chw(raw_image)
            label_array[out_row] = scaler.transform(labels_khz[source_row])

        # flush 确保 memmap 内容落盘。
        image_array.flush()
        label_array.flush()


def count_groups(records: list[SampleRecord], rows: np.ndarray) -> dict[str, int]:
    """统计 split 中三类样本数量。"""
    counts = {group: 0 for group in CASE_GROUPS}
    for row in rows:
        counts[records[int(row)].group] += 1
    return counts


def unique_values_by_chunks(array: np.ndarray, chunk_size: int = 256) -> list[float]:
    """
    分块扫描大图像数组的全部像素值。

    训练图像约 2.1GB，直接 np.unique(train_images) 会一次性触发较大内存压力；
    分块扫描可以完整验证二值性，同时保持内存占用稳定。
    """
    values: set[float] = set()
    for start in range(0, array.shape[0], chunk_size):
        stop = min(start + chunk_size, array.shape[0])
        values.update(float(x) for x in np.unique(array[start:stop]))
    return sorted(values)


def validate_processed_data(output_dir: Path) -> dict[str, object]:
    """验证预处理结果是否满足 prompt 的形状、dtype、取值范围要求。"""
    train_images = np.load(output_dir / "train_images.npy", mmap_mode="r")
    test_images = np.load(output_dir / "test_images.npy", mmap_mode="r")
    train_labels = np.load(output_dir / "train_labels.npy", mmap_mode="r")
    test_labels = np.load(output_dir / "test_labels.npy", mmap_mode="r")

    validations = {
        "train_images_shape": list(train_images.shape),
        "test_images_shape": list(test_images.shape),
        "train_labels_shape": list(train_labels.shape),
        "test_labels_shape": list(test_labels.shape),
        "image_dtype": str(train_images.dtype),
        "label_dtype": str(train_labels.dtype),
        "train_label_min": float(train_labels.min()),
        "train_label_max": float(train_labels.max()),
        "test_label_min": float(test_labels.min()),
        "test_label_max": float(test_labels.max()),
        "train_image_unique_full_scan": unique_values_by_chunks(train_images),
        "test_image_unique_full_scan": unique_values_by_chunks(test_images),
    }

    assert train_images.shape == (8700, 1, IMAGE_SIZE, IMAGE_SIZE)
    assert test_images.shape == (300, 1, IMAGE_SIZE, IMAGE_SIZE)
    assert train_labels.shape == (8700, LABEL_LENGTH)
    assert test_labels.shape == (300, LABEL_LENGTH)
    assert train_images.dtype == np.float32
    assert train_labels.dtype == np.float32
    assert set(validations["train_image_unique_full_scan"]).issubset({0.0, 1.0})
    assert set(validations["test_image_unique_full_scan"]).issubset({0.0, 1.0})
    assert validations["train_label_min"] >= -1e-6
    assert validations["test_label_min"] >= -1e-6
    assert validations["train_label_max"] <= 1.0 + 1e-6
    assert validations["test_label_max"] <= 1.0 + 1e-6

    train_loader, test_loader = build_dataloaders(output_dir, batch_size=BATCH_SIZE)
    train_x, train_y = next(iter(train_loader))
    test_x, test_y = next(iter(test_loader))
    validations.update(
        {
            "train_batch_x_shape": list(train_x.shape),
            "train_batch_y_shape": list(train_y.shape),
            "test_batch_x_shape": list(test_x.shape),
            "test_batch_y_shape": list(test_y.shape),
            "torch_image_dtype": str(train_x.dtype),
            "torch_label_dtype": str(train_y.dtype),
            "cuda_available": bool(torch.cuda.is_available()),
        }
    )

    assert train_x.shape[1:] == (1, IMAGE_SIZE, IMAGE_SIZE)
    assert train_y.shape[1:] == (LABEL_LENGTH,)
    assert test_x.shape[1:] == (1, IMAGE_SIZE, IMAGE_SIZE)
    assert test_y.shape[1:] == (LABEL_LENGTH,)
    assert train_x.dtype == torch.float32
    assert train_y.dtype == torch.float32
    return validations


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def write_report(
    output_dir: Path,
    metadata: dict[str, object],
    validations: dict[str, object],
) -> None:
    """写 Markdown 报告，作为人工可读的预处理留档。"""
    report_path = output_dir / "preprocess_report.md"
    files = [
        "train_images.npy",
        "train_labels.npy",
        "test_images.npy",
        "test_labels.npy",
        "frequency_minmax_scaler.npz",
        "split_indices.npz",
        "split_manifest.csv",
        "metadata.json",
        "preprocess.log",
    ]

    lines = [
        "# Phononic Dataset Preprocessing Report",
        "",
        "## 处理目标",
        "",
        "- 论文样本数：9000",
        "- 输入图像：左上角 256x256，保存为 `(1, 256, 256)` 的 `float32` 二值张量",
        "- 输出标签：1464 点频率序列，单位 kHz，Min-Max 归一化到 `[0, 1]`",
        "- 训练/测试：每类 2900/100，合计 8700/300",
        "",
        "## 关键参数",
        "",
        "```json",
        json.dumps(metadata["config"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## 归一化参数",
        "",
        "```json",
        json.dumps(metadata["scaler"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## 划分统计",
        "",
        "```json",
        json.dumps(metadata["split_counts"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## 验证结果",
        "",
        "```json",
        json.dumps(validations, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 输出文件",
        "",
    ]

    for name in files:
        path = output_dir / name
        if path.exists():
            lines.append(f"- `{name}`: {file_size_mb(path):.2f} MB")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="预处理声子晶体 9000 样本数据集。")
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("processed_phononic_9000"))
    parser.add_argument("--source", choices=("auto", "files", "npy"), default="auto")
    parser.add_argument("--images-npy", type=str, default="images.npy")
    parser.add_argument("--labels-npy", type=str, default="labels.npy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--frequency-key", type=str, default="F")
    parser.add_argument("--frequency-scale", type=float, default=MAT_FREQUENCY_SCALE_TO_KHZ)
    parser.add_argument("--frequency-unit", type=str, default="kHz")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    """准备输出目录；默认不覆盖，防止误删已有处理结果。"""
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"{output_dir} already exists. Use --overwrite to regenerate it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    output_dir = (data_root / args.output_dir).resolve()

    prepare_output_dir(output_dir, overwrite=args.overwrite)
    setup_logging(output_dir)

    logging.info("开始预处理声子晶体论文 9000 样本数据集")
    logging.info("data_root=%s", data_root)
    logging.info("output_dir=%s", output_dir)

    records = build_paper_records()
    train_indices, test_indices = split_records(records, seed=args.seed)

    images_path = data_root / args.images_npy
    labels_path = data_root / args.labels_npy
    if args.source == "auto":
        source = "npy" if images_path.exists() and labels_path.exists() else "files"
    else:
        source = args.source

    logging.info("数据源类型：%s", source)
    logging.info("训练集数量：%d，测试集数量：%d", len(train_indices), len(test_indices))
    logging.info("训练集类别统计：%s", count_groups(records, train_indices))
    logging.info("测试集类别统计：%s", count_groups(records, test_indices))

    source_images = None
    source_indices = None
    if source == "npy":
        logging.info("加载 images.npy / labels.npy 源数据")
        source_images, labels_khz, source_indices = load_packed_npy_source(
            data_root,
            images_name=args.images_npy,
            labels_name=args.labels_npy,
        )
        logging.info(".npy 标签默认按已给单位处理；如需换单位，请调整 --frequency-scale")
        labels_khz = labels_khz * np.float32(args.frequency_scale)
    else:
        logging.info("加载 inputs/*.png 和 outputs/*.mat 源数据")
        labels_khz = load_file_labels(
            data_root=data_root,
            records=records,
            frequency_key=args.frequency_key,
            frequency_scale=args.frequency_scale,
        )

    scaler = FrequencyMinMaxScaler.fit(labels_khz, unit=args.frequency_unit)
    scaler.save(output_dir / "frequency_minmax_scaler.npz")
    logging.info("频率 Min-Max：min=%.8g %s, max=%.8g %s", scaler.min_value, scaler.unit, scaler.max_value, scaler.unit)

    np.savez(
        output_dir / "split_indices.npz",
        train_indices=train_indices,
        test_indices=test_indices,
    )
    write_split_manifest(output_dir / "split_manifest.csv", records, train_indices, test_indices)

    write_processed_arrays(
        source=source,
        data_root=data_root,
        output_dir=output_dir,
        records=records,
        labels_khz=labels_khz,
        scaler=scaler,
        train_indices=train_indices,
        test_indices=test_indices,
        source_images=source_images,
        source_indices=source_indices,
    )

    validations = validate_processed_data(output_dir)
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "data_root": str(data_root),
            "source": source,
            "output_dir": str(output_dir),
            "seed": args.seed,
            "batch_size": args.batch_size,
            "image_shape": [1, IMAGE_SIZE, IMAGE_SIZE],
            "label_length": LABEL_LENGTH,
            "frequency_key": args.frequency_key,
            "frequency_scale": args.frequency_scale,
            "frequency_unit": args.frequency_unit,
            "normalization": "global Min-Max fitted on all 9000 paper samples",
        },
        "scaler": scaler.to_dict(),
        "split_counts": {
            "train_total": int(len(train_indices)),
            "test_total": int(len(test_indices)),
            "train_by_group": count_groups(records, train_indices),
            "test_by_group": count_groups(records, test_indices),
        },
        "validations": validations,
    }

    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(output_dir, metadata, validations)

    logging.info("预处理完成")
    logging.info("输出目录：%s", output_dir)
    logging.info("DataLoader 验证 train batch X=%s, Y=%s", validations["train_batch_x_shape"], validations["train_batch_y_shape"])


if __name__ == "__main__":
    main()
