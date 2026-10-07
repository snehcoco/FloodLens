"""U-Net training, evaluation, and Monte Carlo Dropout inference helpers."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np


def validate_event_split(
    train_events: Iterable[str],
    validation_events: Iterable[str],
    test_events: Iterable[str],
    forbidden_event: str = "Trishuli",
) -> None:
    """Reject event leakage between splits and keep the case study unseen."""
    train, validation, test = map(set, (train_events, validation_events, test_events))
    if train & validation or train & test or validation & test:
        raise ValueError("Train, validation, and test event IDs must be disjoint.")
    all_events = train | validation | test
    if any(forbidden_event.casefold() in event.casefold() for event in all_events):
        raise ValueError(f"{forbidden_event} must remain unseen during training and model selection.")


def build_unet(pretrained: bool = True) -> Any:
    """Build an EfficientNet-B0 U-Net for two-band Sentinel-1 inputs."""
    try:
        import segmentation_models_pytorch as smp
        import torch
        from torch.nn import functional as functional
    except ImportError as exc:
        raise RuntimeError(
            "The segmentation model requires torch and segmentation-models-pytorch; install requirements.txt."
        ) from exc
    model = smp.Unet(
        encoder_name="efficientnet-b0",
        encoder_weights="imagenet" if pretrained else None,
        in_channels=2,
        classes=1,
        activation=None,
    )
    model._floodlens_mc_dropout_enabled = False

    def dropout_hook(module: Any, inputs: tuple[Any, ...], output: Any) -> Any:
        if getattr(model, "_floodlens_mc_dropout_enabled", False):
            return functional.dropout2d(output, p=0.1, training=True)
        return output

    model._floodlens_dropout_handles = [
        block.register_forward_hook(dropout_hook)
        for block in model.decoder.blocks
    ]
    return model


def train_unet(
    model: Any,
    train_loader: Any,
    validation_loader: Any,
    *,
    epochs: int = 20,
    learning_rate: float = 1e-4,
    device: str | None = None,
    checkpoint_path: str | Path | None = None,
) -> list[dict[str, float]]:
    """Train on supplied event-separated VV/VH loaders using BCE + Dice loss."""
    try:
        import segmentation_models_pytorch as smp
        import torch
    except ImportError as exc:
        raise RuntimeError("Training requires torch and segmentation-models-pytorch.") from exc
    if epochs < 1:
        raise ValueError("epochs must be at least 1.")
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(target_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    bce = torch.nn.BCEWithLogitsLoss()
    dice = smp.losses.DiceLoss(mode="binary", from_logits=True)
    history: list[dict[str, float]] = []
    best_validation_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_losses: list[float] = []
        for images, masks in train_loader:
            images, masks = images.to(target_device), masks.to(target_device).float()
            if images.ndim != 4 or images.shape[1] != 2:
                raise ValueError("Each SAR batch must have shape (batch, 2, height, width).")
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = 0.5 * bce(logits, masks) + 0.5 * dice(logits, masks)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        validation_losses: list[float] = []
        with torch.no_grad():
            for images, masks in validation_loader:
                images, masks = images.to(target_device), masks.to(target_device).float()
                logits = model(images)
                loss = 0.5 * bce(logits, masks) + 0.5 * dice(logits, masks)
                validation_losses.append(float(loss.cpu()))
        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        validation_loss = float(np.mean(validation_losses)) if validation_losses else float("nan")
        history.append({"epoch": float(epoch + 1), "train_loss": train_loss, "validation_loss": validation_loss})
        if checkpoint_path is not None and validation_losses and validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            destination = Path(checkpoint_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), destination)
    return history


def mc_dropout_predict(model: Any, image: Any, passes: int = 20, device: str | None = None) -> tuple[Any, Any]:
    """Return mean probability and standard-deviation maps without training BatchNorm."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Monte Carlo inference requires torch.") from exc
    if passes < 2:
        raise ValueError("At least two stochastic passes are required to estimate uncertainty.")
    target_device = torch.device(device or next(model.parameters()).device)
    model.eval()
    model._floodlens_mc_dropout_enabled = True
    samples = []
    try:
        with torch.no_grad():
            for _ in range(passes):
                logits = model(image.to(target_device))
                samples.append(torch.sigmoid(logits))
    finally:
        model._floodlens_mc_dropout_enabled = False
        model.eval()
    stacked = torch.stack(samples)
    return stacked.mean(dim=0), stacked.std(dim=0, unbiased=False)


def segmentation_metrics(probability: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Compute binary IoU, F1, precision, recall, and expected calibration error."""
    probs = np.asarray(probability, dtype=np.float64).ravel()
    truth = np.asarray(target, dtype=bool).ravel()
    if probs.shape != truth.shape:
        raise ValueError("Probability and target arrays must have the same shape.")
    valid = np.isfinite(probs)
    probs = np.clip(probs[valid], 0, 1)
    truth = truth[valid]
    pred = probs >= threshold
    tp = int(np.count_nonzero(pred & truth))
    fp = int(np.count_nonzero(pred & ~truth))
    fn = int(np.count_nonzero(~pred & truth))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    union = tp + fp + fn
    ece = 0.0
    if probs.size:
        for lower in np.linspace(0, 1, 11)[:-1]:
            upper = lower + 0.1
            in_bin = (probs >= lower) & (probs < upper if upper < 1 else probs <= upper)
            if in_bin.any():
                ece += float(in_bin.mean() * abs(probs[in_bin].mean() - truth[in_bin].mean()))
    return {
        "iou": tp / union if union else 0.0,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "ece": ece,
    }


def save_uncertainty_raster(
    uncertainty: np.ndarray,
    reference_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Save pixel uncertainty using the CRS and grid of a reference raster."""
    import rasterio

    output = Path(output_path)
    with rasterio.open(reference_path) as reference:
        if uncertainty.shape[-2:] != (reference.height, reference.width):
            raise ValueError("Uncertainty array does not match the reference raster grid.")
        profile = reference.profile.copy()
        profile.update(count=1, dtype="float32", compress="deflate")
    output.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output, "w", **profile) as destination:
        destination.write(np.asarray(uncertainty).squeeze().astype(np.float32), 1)
    return output


def save_model_outputs(
    probability: Any,
    uncertainty: Any,
    reference_path: str | Path,
    output_dir: str | Path,
    threshold: float = 0.5,
) -> dict[str, Path]:
    """Export probability, binary prediction, and MC-dropout uncertainty GeoTIFFs."""
    import rasterio

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    probability_array = np.asarray(probability.detach().cpu() if hasattr(probability, "detach") else probability).squeeze()
    uncertainty_array = np.asarray(uncertainty.detach().cpu() if hasattr(uncertainty, "detach") else uncertainty).squeeze()
    if probability_array.ndim != 2 or uncertainty_array.shape != probability_array.shape:
        raise ValueError("Probability and uncertainty outputs must be same-sized 2D arrays.")
    with rasterio.open(reference_path) as reference:
        if probability_array.shape != (reference.height, reference.width):
            raise ValueError("Model outputs do not match the reference raster grid.")
        profile = reference.profile.copy()
        profile.update(count=1, dtype="float32", compress="deflate")
    products = {
        "probability": output / "predicted_probability.tif",
        "mask": output / "predicted_mask.tif",
        "uncertainty": output / "uncertainty_map.tif",
    }
    for key, array in (
        ("probability", probability_array.astype(np.float32)),
        ("mask", (probability_array >= threshold).astype(np.float32)),
        ("uncertainty", uncertainty_array.astype(np.float32)),
    ):
        with rasterio.open(products[key], "w", **profile) as destination:
            destination.write(array, 1)
    return products
