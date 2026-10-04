"""Train/evaluate the contractive model on BSDS500 for known Gaussian blur."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOCAL_DEPS = ROOT / "data" / "pydeps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import DataLoader, Dataset

from linear_inverse import ContractiveInverseNet


class GaussianBlurOperator:
    """Fixed 9x9, sigma-2 RGB blur with periodic boundary conditions."""

    def __init__(self, image_size=64, device="cpu", kernel_size=9, sigma=2.0):
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd")
        coords = torch.arange(kernel_size, dtype=torch.float64) - kernel_size // 2
        one_d = torch.exp(-(coords**2) / (2 * sigma**2))
        one_d /= one_d.sum()
        kernel = torch.outer(one_d, one_d).float()
        self.radius = kernel_size // 2
        self.kernel = kernel[None, None].repeat(3, 1, 1, 1).to(device)

        # For circular convolution, the DFT diagonalizes A exactly.
        response_grid = torch.zeros((image_size, image_size), dtype=torch.float64)
        for iy in range(kernel_size):
            for ix in range(kernel_size):
                dy, dx = iy - self.radius, ix - self.radius
                response_grid[dy % image_size, dx % image_size] = kernel[iy, ix]
        magnitudes = torch.fft.fft2(response_grid).abs()
        self.sigma_max = float(magnitudes.max())
        self.sigma_min = float(magnitudes.min())

    def forward(self, x):
        x = F.pad(x, (self.radius,) * 4, mode="circular")
        return F.conv2d(x, self.kernel, groups=3)

    def adjoint(self, y):
        # The Gaussian kernel is symmetric, so A^T = A under periodic borders.
        return self.forward(y)


class RandomPatchDataset(Dataset):
    def __init__(self, image_dir, patch_size=64, samples=1024, seed=0,
                 augment=False):
        self.paths = sorted(Path(image_dir).glob("*.jpg"))
        if not self.paths:
            raise FileNotFoundError(f"No .jpg images found under {image_dir}")
        self.patch_size = patch_size
        self.samples = samples
        self.seed = seed
        self.epoch = 0
        self.augment = augment

    def __len__(self):
        return self.samples

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        rng = random.Random(self.seed + self.epoch * self.samples + index)
        # Evaluation walks the split in stable order so a 200-patch BSD500
        # test uses every one of its 200 held-out images exactly once.
        path = (self.paths[index % len(self.paths)] if not self.augment
                else self.paths[rng.randrange(len(self.paths))])
        with Image.open(path) as image:
            image = image.convert("RGB")
            width, height = image.size
            p = self.patch_size
            if width < p or height < p:
                raise ValueError(f"Image {path} is smaller than {p}x{p}")
            left = rng.randrange(width - p + 1)
            top = rng.randrange(height - p + 1)
            patch = np.asarray(image.crop((left, top, left + p, top + p)),
                               dtype=np.float32) / 255.0
        tensor = torch.from_numpy(patch.copy()).permute(2, 0, 1)
        if self.augment:
            if rng.random() < 0.5:
                tensor = tensor.flip(-1)
            if rng.random() < 0.5:
                tensor = tensor.flip(-2)
            tensor = torch.rot90(tensor, rng.randrange(4), (-2, -1))
        return tensor


@torch.no_grad()
def evaluate(model, operator, image_dir, *, samples=200, batch_size=4,
             noise_std=5 / 255, seed=1234, device="cpu"):
    dataset = RandomPatchDataset(image_dir, samples=samples, seed=seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0)
    model.eval()
    totals = {"backprojection_psnr": [], "restored_psnr": [],
              "backprojection_ssim": [], "restored_ssim": []}
    rng = torch.Generator(device="cpu").manual_seed(seed)
    for clean in loader:
        clean = clean.to(device)
        measurement = operator.forward(clean)
        noise = torch.randn(measurement.shape, generator=rng,
                            dtype=measurement.dtype).to(device) * noise_std
        measurement = measurement + noise
        initial = operator.adjoint(measurement)
        restored = model(initial, measurement, operator).clamp(0, 1)
        clean_np = clean.cpu().permute(0, 2, 3, 1).numpy()
        noisy_np = initial.clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
        restored_np = restored.cpu().permute(0, 2, 3, 1).numpy()
        for ref, baseline, estimate in zip(clean_np, noisy_np, restored_np):
            totals["backprojection_psnr"].append(peak_signal_noise_ratio(
                ref, baseline, data_range=1.0))
            totals["restored_psnr"].append(peak_signal_noise_ratio(ref, estimate,
                                                                   data_range=1.0))
            totals["backprojection_ssim"].append(structural_similarity(
                ref, baseline, data_range=1.0, channel_axis=-1))
            totals["restored_ssim"].append(structural_similarity(
                ref, estimate, data_range=1.0, channel_axis=-1))
    model.train()
    return {key: float(np.mean(values)) for key, values in totals.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path,
                        default=ROOT / "data" / "BSDS500" / "BSR" /
                                "BSDS500" / "data" / "images")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--samples-per-epoch", type=int, default=1024)
    parser.add_argument("--validation-samples", type=int, default=100)
    parser.add_argument("--test-samples", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--depth", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--noise-std", type=float, default=5 / 255)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" /
                        "bsd500_gaussian_blur")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(max(1, min(8, torch.get_num_threads())))

    for split in ("train", "val", "test"):
        if not (args.data_root / split).is_dir():
            raise FileNotFoundError(f"Missing BSD500 split: {args.data_root / split}")

    operator = GaussianBlurOperator(device=device)
    model = ContractiveInverseNet(
        image_shape=(64, 64), sigma_max=operator.sigma_max,
        sigma_min=operator.sigma_min, depth=args.depth, levels=3, channels=3
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    train_data = RandomPatchDataset(args.data_root / "train", samples=args.samples_per_epoch,
                                    seed=args.seed, augment=True)
    train_loader = DataLoader(train_data, batch_size=args.batch_size,
                              shuffle=False, num_workers=0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_val = -float("inf")
    best_epoch = 0
    history = []

    print(f"Device: {device}; A: Gaussian blur 9x9, sigma=2; AWGN sigma={args.noise_std:.6g}")
    print(f"Singular values: sigma_max={operator.sigma_max:.8g}, "
          f"sigma_min={operator.sigma_min:.8g}; "
          f"gradient step 0 < gamma < {2 / operator.sigma_max**2:.8g}")
    for epoch in range(args.epochs):
        train_data.set_epoch(epoch)
        model.train()
        losses = []
        for clean in train_loader:
            clean = clean.to(device)
            measurement = operator.forward(clean)
            measurement = measurement + torch.randn_like(measurement) * args.noise_std
            initial = operator.adjoint(measurement)
            estimate = model(initial, measurement, operator)
            loss = F.mse_loss(estimate, clean)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        if epoch in {max(0, args.epochs // 3 - 1), max(0, 2 * args.epochs // 3 - 1)}:
            for group in optimizer.param_groups:
                group["lr"] *= 0.5
        val = evaluate(model, operator, args.data_root / "val",
                       samples=args.validation_samples, batch_size=args.batch_size,
                       noise_std=args.noise_std, seed=args.seed + 10000, device=device)
        row = {"epoch": epoch + 1, "train_mse": float(np.mean(losses)), **val}
        history.append(row)
        print(json.dumps(row))
        if val["restored_psnr"] > best_val:
            best_val = val["restored_psnr"]
            best_epoch = epoch + 1
            torch.save(model.state_dict(), args.output_dir / "best_model.pth")
        with open(args.output_dir / "training_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    checkpoint = args.output_dir / "best_model.pth"
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    test = evaluate(model, operator, args.data_root / "test",
                    samples=args.test_samples, batch_size=args.batch_size,
                    noise_std=args.noise_std, seed=args.seed + 20000, device=device)
    learned_steps = [float(layer.step_size.detach().cpu()) for layer in model.layers]
    gradient_bounds = [float(layer.gradient_bound().detach().cpu())
                       for layer in model.layers]
    network_bound = 1.0
    for layer, q in zip(model.layers, gradient_bounds):
        network_bound *= q / (q + layer.epsilon)
    report = {
        "operator": "periodic 9x9 Gaussian blur, sigma=2",
        "noise_std": args.noise_std,
        "epochs": args.epochs,
        "depth": args.depth,
        "batch_size": args.batch_size,
        "samples_per_epoch": args.samples_per_epoch,
        "best_validation_epoch": best_epoch,
        "best_validation_psnr": best_val,
        "train_images": len(list((args.data_root / "train").glob("*.jpg"))),
        "validation_images": len(list((args.data_root / "val").glob("*.jpg"))),
        "test_images": len(list((args.data_root / "test").glob("*.jpg"))),
        "validation_patch_count": args.validation_samples,
        "test_patch_count": args.test_samples,
        "sigma_max": operator.sigma_max,
        "sigma_min": operator.sigma_min,
        "step_size_upper_bound": 2 / operator.sigma_max**2,
        "learned_step_size_min": min(learned_steps),
        "learned_step_size_max": max(learned_steps),
        "max_gradient_step_lipschitz_bound": max(gradient_bounds),
        "network_lipschitz_upper_bound": network_bound,
        **test,
    }
    with open(args.output_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("FINAL TEST", json.dumps(report))


if __name__ == "__main__":
    main()
