"""
Data loading and preprocessing for the phononic-crystal surrogate dataset.

This script supports two input layouts:
1. Packed NumPy files:
   - images.npy: shape [N, H, W], [N, H, W, C], or [N, C, H, W]
   - labels.npy: shape [N, 1464] or compatible with [N, 1464, 1]
2. The original dataset layout in this folder:
   - inputs/PC_label_{case_id}.png
   - outputs/out_lines_{case_id}_a8_h3.mat

The paper subset contains 9000 cases:
cross-like    : 1001-4000
diagonal-like : 4001-7000
blot-like     : 8001-11000
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


IMAGE_SIZE = 256
LABEL_LENGTH = 1464
CASE_GROUPS = {
    "cross": range(1001, 4001),
    "diagonal": range(4001, 7001),
    "blot": range(8001, 11001),
}


@dataclass(frozen=True)
class FrequencyMinMaxScaler:
    """Global Min-Max scaler for frequency vectors."""

    min_value: float
    max_value: float

    @property
    def scale(self) -> float:
        return max(self.max_value - self.min_value, 1e-12)

    @classmethod
    def fit(cls, values: np.ndarray) -> "FrequencyMinMaxScaler":
        values = np.asarray(values, dtype=np.float32)
        return cls(min_value=float(values.min()), max_value=float(values.max()))

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        return ((values - self.min_value) / self.scale).astype(np.float32)

    def inverse_transform(
        self, values: np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        """Convert normalized frequencies back to the original unit."""
        if torch.is_tensor(values):
            return values * self.scale + self.min_value

        values = np.asarray(values, dtype=np.float32)
        return (values * self.scale + self.min_value).astype(np.float32)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            min_value=np.array(self.min_value, dtype=np.float32),
            max_value=np.array(self.max_value, dtype=np.float32),
        )

    @classmethod
    def load(cls, path: str | Path) -> "FrequencyMinMaxScaler":
        data = np.load(path)
        return cls(
            min_value=float(data["min_value"]),
            max_value=float(data["max_value"]),
        )


def build_paper_case_index() -> tuple[np.ndarray, np.ndarray]:
    """Return case ids and shape-group labels for the 9000 paper cases."""
    case_ids: list[int] = []
    groups: list[str] = []

    for group_name, id_range in CASE_GROUPS.items():
        ids = list(id_range)
        case_ids.extend(ids)
        groups.extend([group_name] * len(ids))

    return np.asarray(case_ids, dtype=np.int64), np.asarray(groups)


def infer_npy_paper_indices(num_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Infer paper-case indices for packed .npy files.

    If N == 9000, the arrays are assumed to already contain the paper subset
    ordered as cross, diagonal, blot.
    If N == 11000, the arrays are assumed to be ordered by case id 1..11000,
    so the paper subset is selected by the README ranges.
    """
    case_ids, groups = build_paper_case_index()

    if num_samples == 9000:
        return np.arange(9000, dtype=np.int64), groups

    if num_samples == 11000:
        return case_ids - 1, groups

    raise ValueError(
        f"Expected 9000 paper samples or 11000 full samples, got {num_samples}."
    )


def stratified_train_test_split(
    groups: Iterable[str],
    train_per_group: int = 2900,
    test_per_group: int = 100,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Randomly split each shape group into 2900 train and 100 test samples."""
    groups = np.asarray(list(groups))
    rng = np.random.default_rng(seed)
    train_indices: list[np.ndarray] = []
    test_indices: list[np.ndarray] = []

    for group_name in dict.fromkeys(groups):
        group_indices = np.flatnonzero(groups == group_name)
        required = train_per_group + test_per_group
        if group_indices.size < required:
            raise ValueError(
                f"Group {group_name!r} has {group_indices.size} samples, "
                f"but {required} are required."
            )

        shuffled = group_indices.copy()
        rng.shuffle(shuffled)
        train_indices.append(shuffled[:train_per_group])
        test_indices.append(shuffled[train_per_group:required])

    train = np.concatenate(train_indices).astype(np.int64)
    test = np.concatenate(test_indices).astype(np.int64)
    return train, test


def preprocess_image_array(image: np.ndarray, image_size: int = IMAGE_SIZE) -> np.ndarray:
    """
    Crop the top-left 256x256 quarter and return a binary float32 CHW tensor array.

    Output shape is always [1, 256, 256], with values in {0.0, 1.0}.
    """
    arr = np.asarray(image)

    if arr.ndim == 3:
        if arr.shape[0] in (1, 3, 4):  # CHW
            arr = arr[0]
        elif arr.shape[-1] in (1, 3, 4):  # HWC
            arr = arr[..., 0]
        else:
            raise ValueError(f"Cannot infer image channel dimension from {arr.shape}.")
    elif arr.ndim != 2:
        raise ValueError(f"Expected a 2D image or a 3D image with channels, got {arr.shape}.")

    if arr.shape[0] < image_size or arr.shape[1] < image_size:
        raise ValueError(
            f"Image is too small for {image_size}x{image_size} crop: {arr.shape}."
        )

    arr = arr[:image_size, :image_size]
    arr = (arr > 0).astype(np.float32)
    return arr[None, :, :]


def preprocess_labels(labels: np.ndarray) -> np.ndarray:
    """Convert labels to [N, 1464] float32 frequency vectors."""
    labels = np.asarray(labels, dtype=np.float32)
    if labels.ndim < 2:
        raise ValueError(f"Expected label array with sample dimension, got {labels.shape}.")

    labels = labels.reshape(labels.shape[0], -1)
    if labels.shape[1] != LABEL_LENGTH:
        raise ValueError(
            f"Expected labels with length {LABEL_LENGTH}, got shape {labels.shape}."
        )

    return labels.astype(np.float32, copy=False)


def load_png_image(path: Path) -> np.ndarray:
    """Load one input PNG as a grayscale NumPy array."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Please install Pillow to read PNG inputs: pip install pillow") from exc

    with Image.open(path) as image:
        return np.asarray(image.convert("L"))


def load_frequency_from_mat(path: Path, frequency_key: str = "F") -> np.ndarray:
    """
    Load one 1464-point frequency vector from a MATLAB file.

    The provided dataset uses MATLAB 7.3/HDF5 .mat files, so h5py is tried first.
    scipy.io.loadmat is kept as a fallback for older MATLAB .mat files.
    """
    try:
        import h5py
    except ImportError:
        h5py = None

    if h5py is not None:
        try:
            with h5py.File(path, "r") as handle:
                if frequency_key not in handle:
                    raise KeyError(
                        f"{path} does not contain frequency field {frequency_key!r}."
                    )
                values = np.asarray(handle[frequency_key]).reshape(-1)
                values = values.astype(np.float32)
                if values.size != LABEL_LENGTH:
                    raise ValueError(
                        f"{path} has frequency length {values.size}, "
                        f"expected {LABEL_LENGTH}."
                    )
                return values
        except OSError:
            pass

    try:
        from scipy.io import loadmat
    except ImportError as exc:
        raise ImportError(
            "Please install h5py for MATLAB 7.3 files or scipy for older .mat files."
        ) from exc

    mat = loadmat(path)
    if frequency_key not in mat:
        raise KeyError(f"{path} does not contain frequency field {frequency_key!r}.")

    values = np.asarray(mat[frequency_key]).reshape(-1).astype(np.float32)
    if values.size != LABEL_LENGTH:
        raise ValueError(
            f"{path} has frequency length {values.size}, expected {LABEL_LENGTH}."
        )
    return values


def load_file_labels(
    output_dir: Path,
    case_ids: np.ndarray,
    frequency_key: str = "F",
) -> np.ndarray:
    """Load all paper-case frequency vectors from the original .mat files."""
    labels = np.empty((case_ids.size, LABEL_LENGTH), dtype=np.float32)

    for row, case_id in enumerate(case_ids):
        path = output_dir / f"out_lines_{int(case_id)}_a8_h3.mat"
        if not path.exists():
            raise FileNotFoundError(path)
        labels[row] = load_frequency_from_mat(path, frequency_key=frequency_key)

    return labels


class PhononicDataset(Dataset):
    """
    PyTorch Dataset returning:
    image: torch.float32 tensor, shape [1, 256, 256], values 0/1
    label: torch.float32 tensor, shape [1464], normalized to [0, 1]
    """

    def __init__(
        self,
        labels: np.ndarray,
        sample_indices: np.ndarray,
        *,
        images: np.ndarray | None = None,
        image_indices: np.ndarray | None = None,
        case_ids: np.ndarray | None = None,
        input_dir: Path | None = None,
    ) -> None:
        self.labels = preprocess_labels(labels)
        self.sample_indices = np.asarray(sample_indices, dtype=np.int64)
        self.images = images
        self.image_indices = image_indices
        self.case_ids = case_ids
        self.input_dir = input_dir

        if self.images is None:
            if self.case_ids is None or self.input_dir is None:
                raise ValueError("case_ids and input_dir are required when images=None.")
        elif self.image_indices is None:
            self.image_indices = np.arange(len(self.labels), dtype=np.int64)

    def __len__(self) -> int:
        return int(self.sample_indices.size)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = int(self.sample_indices[index])

        if self.images is not None:
            image_row = int(self.image_indices[row])
            raw_image = self.images[image_row]
        else:
            case_id = int(self.case_ids[row])
            raw_image = load_png_image(self.input_dir / f"PC_label_{case_id}.png")

        image = preprocess_image_array(raw_image)
        label = self.labels[row]

        return torch.from_numpy(image).float(), torch.from_numpy(label).float()


def make_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def create_dataloaders_from_npy(
    images_npy: Path,
    labels_npy: Path,
    *,
    batch_size: int = 32,
    seed: int = 42,
    num_workers: int = 0,
    scaler_fit: Literal["all", "train"] = "all",
    scaler_path: Path | None = None,
) -> tuple[DataLoader, DataLoader, FrequencyMinMaxScaler]:
    """Create train/test dataloaders from packed images.npy and labels.npy."""
    images = np.load(images_npy, mmap_mode="r")
    raw_labels = preprocess_labels(np.load(labels_npy))

    if len(images) != len(raw_labels):
        raise ValueError(f"Image count {len(images)} != label count {len(raw_labels)}.")

    image_indices, groups = infer_npy_paper_indices(len(raw_labels))
    labels = raw_labels[image_indices]

    train_indices, test_indices = stratified_train_test_split(groups, seed=seed)

    scaler_source = labels if scaler_fit == "all" else labels[train_indices]
    scaler = FrequencyMinMaxScaler.fit(scaler_source)
    labels_norm = scaler.transform(labels)

    if scaler_path is not None:
        scaler.save(scaler_path)

    train_dataset = PhononicDataset(
        labels_norm,
        train_indices,
        images=images,
        image_indices=image_indices,
    )
    test_dataset = PhononicDataset(
        labels_norm,
        test_indices,
        images=images,
        image_indices=image_indices,
    )

    train_loader = make_dataloader(train_dataset, batch_size, True, num_workers)
    test_loader = make_dataloader(test_dataset, batch_size, False, num_workers)
    return train_loader, test_loader, scaler


def create_dataloaders_from_files(
    data_root: Path,
    *,
    batch_size: int = 32,
    seed: int = 42,
    num_workers: int = 0,
    scaler_fit: Literal["all", "train"] = "all",
    scaler_path: Path | None = None,
    frequency_key: str = "F",
) -> tuple[DataLoader, DataLoader, FrequencyMinMaxScaler]:
    """Create train/test dataloaders from inputs/*.png and outputs/*.mat."""
    input_dir = data_root / "inputs"
    output_dir = data_root / "outputs"
    case_ids, groups = build_paper_case_index()

    if not input_dir.exists():
        raise FileNotFoundError(input_dir)
    if not output_dir.exists():
        raise FileNotFoundError(output_dir)

    labels = load_file_labels(output_dir, case_ids, frequency_key=frequency_key)
    train_indices, test_indices = stratified_train_test_split(groups, seed=seed)

    scaler_source = labels if scaler_fit == "all" else labels[train_indices]
    scaler = FrequencyMinMaxScaler.fit(scaler_source)
    labels_norm = scaler.transform(labels)

    if scaler_path is not None:
        scaler.save(scaler_path)

    train_dataset = PhononicDataset(
        labels_norm,
        train_indices,
        case_ids=case_ids,
        input_dir=input_dir,
    )
    test_dataset = PhononicDataset(
        labels_norm,
        test_indices,
        case_ids=case_ids,
        input_dir=input_dir,
    )

    train_loader = make_dataloader(train_dataset, batch_size, True, num_workers)
    test_loader = make_dataloader(test_dataset, batch_size, False, num_workers)
    return train_loader, test_loader, scaler


def validate_one_batch(train_loader: DataLoader, test_loader: DataLoader) -> None:
    """Small runtime sanity check for tensor shape, dtype, and split sizes."""
    train_x, train_y = next(iter(train_loader))
    test_x, test_y = next(iter(test_loader))

    assert len(train_loader.dataset) == 8700
    assert len(test_loader.dataset) == 300
    assert train_x.shape[1:] == (1, IMAGE_SIZE, IMAGE_SIZE)
    assert test_x.shape[1:] == (1, IMAGE_SIZE, IMAGE_SIZE)
    assert train_y.shape[1:] == (LABEL_LENGTH,)
    assert test_y.shape[1:] == (LABEL_LENGTH,)
    assert train_x.dtype == torch.float32
    assert train_y.dtype == torch.float32
    assert set(torch.unique(train_x).cpu().tolist()).issubset({0.0, 1.0})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build PyTorch DataLoaders for phononic-crystal data."
    )
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument(
        "--source",
        choices=("auto", "npy", "files"),
        default="auto",
        help="auto uses images.npy/labels.npy if present, otherwise inputs/outputs files.",
    )
    parser.add_argument("--images-npy", type=Path, default=Path("images.npy"))
    parser.add_argument("--labels-npy", type=Path, default=Path("labels.npy"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--scaler-fit",
        choices=("all", "train"),
        default="all",
        help="Use 'all' to keep all 9000 normalized labels within [0, 1].",
    )
    parser.add_argument(
        "--scaler-path",
        type=Path,
        default=Path("frequency_minmax_scaler.npz"),
        help="Where to save Min-Max normalization parameters.",
    )
    parser.add_argument(
        "--frequency-key",
        type=str,
        default="F",
        help="MATLAB variable name containing the 1464-point frequency vector.",
    )
    parser.add_argument(
        "--skip-batch-check",
        action="store_true",
        help="Skip loading a sample batch after constructing DataLoaders.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    images_npy = (data_root / args.images_npy).resolve()
    labels_npy = (data_root / args.labels_npy).resolve()
    scaler_path = (data_root / args.scaler_path).resolve()

    use_npy = args.source == "npy" or (
        args.source == "auto" and images_npy.exists() and labels_npy.exists()
    )

    if use_npy:
        train_loader, test_loader, scaler = create_dataloaders_from_npy(
            images_npy,
            labels_npy,
            batch_size=args.batch_size,
            seed=args.seed,
            num_workers=args.num_workers,
            scaler_fit=args.scaler_fit,
            scaler_path=scaler_path,
        )
        source_name = "npy"
    else:
        train_loader, test_loader, scaler = create_dataloaders_from_files(
            data_root,
            batch_size=args.batch_size,
            seed=args.seed,
            num_workers=args.num_workers,
            scaler_fit=args.scaler_fit,
            scaler_path=scaler_path,
            frequency_key=args.frequency_key,
        )
        source_name = "files"

    if not args.skip_batch_check:
        validate_one_batch(train_loader, test_loader)

    print(f"Source              : {source_name}")
    print(f"Train samples       : {len(train_loader.dataset)}")
    print(f"Test samples        : {len(test_loader.dataset)}")
    print(f"Batch size          : {args.batch_size}")
    print(f"Frequency min       : {scaler.min_value:.6g}")
    print(f"Frequency max       : {scaler.max_value:.6g}")
    print(f"Scaler saved to     : {scaler_path}")
    print("Example denorm usage: real_freq = scaler.inverse_transform(pred_norm)")


if __name__ == "__main__":
    main()
