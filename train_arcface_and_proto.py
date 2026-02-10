# train_arcface_and_proto_fixed.py
import os
import json
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import timm
from torch.optim.lr_scheduler import CosineAnnealingLR

# =========================
# CONFIG
# =========================
DATA_DIR = r"C:\Identification dataset\KNP_identification_dataset_nighttime"

BATCH_SIZE = 16
NUM_EPOCHS = 30
EARLY_STOP_PATIENCE = 8
EARLY_STOP_MIN_DELTA = 1e-4
NUM_WORKERS = min(4, os.cpu_count() or 1)

SAVE_ROOT = Path("identification_models_KNP") / "KNP_CONVNEXT_ARCFACE_22_Jan_2026_fixed"

# Speed / stability
USE_AMP = True                    # mixed precision on CUDA
GRAD_CLIP_NORM = 1.0              # helps stability
EVAL_EVERY = 1                    # prototype eval frequency (epochs)
BUILD_PROTOS_EVERY = 1            # prototype rebuild frequency (epochs)
WARMUP_FREEZE_EPOCHS = 2          # freeze backbone for first N epochs (optional but often helps)

# =========================
# PATH HELPERS
# =========================
def increment_path(path: Path, mkdir=True) -> Path:
    if not path.exists():
        if mkdir:
            path.mkdir(parents=True, exist_ok=True)
        return path
    i = 1
    while True:
        p = Path(f"{path}_{i:02d}")
        if not p.exists():
            if mkdir:
                p.mkdir(parents=True, exist_ok=True)
            return p
        i += 1

SAVE_PATH = increment_path(SAVE_ROOT)
MODEL_SAVE_PATH = SAVE_PATH / "best_backbone.pth"
LAST_MODEL_SAVE_PATH = SAVE_PATH / "last_backbone.pth"
LABEL_JSON_PATH = SAVE_PATH / "class_mappings.json"
BEST_PROTOS_PATH = SAVE_PATH / "best_prototypes.pt"
LAST_PROTOS_PATH = SAVE_PATH / "last_prototypes.pt"

# =========================
# TRANSFORMS
# =========================
# NOTE: RandomRotation(0..180) is very aggressive for ID; use small rotation unless you have strong reason.
train_transform = transforms.Compose([
#    transforms.RandomRotation(degrees=15),
        transforms.RandomApply(
            [transforms.RandomRotation((0, 180))],
            p=0.15
        ),
        transforms.RandomRotation(15),

    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# For prototype building: no augmentation, deterministic
proto_transform = val_transform


# Add this helper near the top (after imports / config)
class EarlyStopping:
    """
    Early stop on a metric that should be maximized (e.g., Val Proto Acc).
    Stops when no improvement for `patience` eval points.
    """
    def __init__(self, patience=8, min_delta=1e-4):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best = None
        self.bad_count = 0

    def step(self, metric_value: float) -> bool:
        """
        Returns True if training should stop.
        """
        if metric_value != metric_value:  # NaN guard
            return False

        if self.best is None or metric_value > self.best + self.min_delta:
            self.best = metric_value
            self.bad_count = 0
            return False

        self.bad_count += 1
        return self.bad_count >= self.patience

# =========================
# ARCFACE LAYER
# =========================
class ArcMarginProduct(nn.Module):
    """
    Standard ArcFace head (training only)
    """
    def __init__(self, in_features, out_features, s=30.0, m=0.5):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.s = float(s)
        self.m = float(m)

        # register buffers so they move with .to(device)
        self.register_buffer("cos_m", torch.cos(torch.tensor(self.m)))
        self.register_buffer("sin_m", torch.sin(torch.tensor(self.m)))
        self.register_buffer("th",    torch.cos(torch.pi - torch.tensor(self.m)))
        self.register_buffer("mm",    torch.sin(torch.pi - torch.tensor(self.m)) * self.m)

    def forward(self, emb, label):
        emb = F.normalize(emb, dim=1)
        W = F.normalize(self.weight, dim=1)

        cosine = F.linear(emb, W)  # [B, C]
        sine = torch.sqrt(torch.clamp(1.0 - cosine ** 2, min=1e-6))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)

        logits = one_hot * phi + (1.0 - one_hot) * cosine
        logits *= self.s
        return logits

# =========================
# EMBEDDING UTILS
# =========================
@torch.no_grad()
def get_embeddings(model, x):
    feats = model.forward_features(x)
    emb = model.forward_head(feats, pre_logits=True)
    return F.normalize(emb, dim=1)

@torch.no_grad()
def aggregate_embeddings(embeddings_batch, method='mean'):
    """
    Aggregate multiple embeddings into a single robust embedding.
    Better than using max prediction alone for nighttime identification.
    
    Args:
        embeddings_batch: [B, D] tensor of normalized embeddings
        method: 'mean' (average), 'median', or 'first' (fallback)
    
    Returns:
        Single [D] normalized embedding
    """
    if embeddings_batch.shape[0] == 0:
        raise ValueError("Empty embedding batch")
    
    if method == 'mean':
        agg_emb = embeddings_batch.mean(dim=0)
    elif method == 'median':
        agg_emb = embeddings_batch.median(dim=0).values
    else:  # fallback to first
        agg_emb = embeddings_batch[0]
    
    return F.normalize(agg_emb.unsqueeze(0), dim=1).squeeze(0)

@torch.no_grad()
def fuse_modalities(ir_embedding, rgb_embedding, alpha=0.5):
    """
    Fuse IR and RGB embeddings for multi-modal identification.
    
    Args:
        ir_embedding: [D] normalized IR embedding
        rgb_embedding: [D] normalized RGB embedding
        alpha: weight for IR (1-alpha for RGB)
    
    Returns:
        Fused [D] normalized embedding
    """
    fused = alpha * ir_embedding + (1 - alpha) * rgb_embedding
    return F.normalize(fused.unsqueeze(0), dim=1).squeeze(0)

# =========================
# PROTOTYPES (FAST, SAFE)
# =========================
@torch.no_grad()
def build_class_prototypes_from_loader(model, loader, num_classes, device):
    """
    Builds prototypes by summing embeddings per class and dividing by count.
    This is faster and safer than storing per-image embeddings in lists.
    """
    model.eval()

    # infer embedding dim from one batch
    emb_dim = None
    sums = None
    counts = torch.zeros(num_classes, dtype=torch.long)

    for x, y in tqdm(loader, desc="Build Prototypes", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        emb = get_embeddings(model, x)  # [B, D]
        if emb_dim is None:
            emb_dim = emb.shape[1]
            sums = torch.zeros(num_classes, emb_dim, device=device)

        # accumulate
        for cls in y.unique():
            cls = int(cls.item())
            mask = (y == cls)
            sums[cls] += emb[mask].sum(dim=0)
            counts[cls] += int(mask.sum().item())

    # build prototypes (handle empty class safely)
    protos = torch.zeros(num_classes, sums.shape[1], device=device)
    for c in range(num_classes):
        if counts[c] > 0:
            protos[c] = sums[c] / counts[c].float()
    protos = F.normalize(protos, dim=1)

    return protos, counts.cpu()

@torch.no_grad()
def prototype_accuracy(model, val_loader, protos, device):
    """
    Proper metric:
      pred = argmax cosine_similarity(emb, prototypes)
    """
    model.eval()
    correct = 0
    total = 0

    for x, y in tqdm(val_loader, desc="Val (Proto Acc)", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        emb = get_embeddings(model, x)           # [B, D]
        sims = emb @ protos.t()                 # [B, C]
        pred = sims.argmax(dim=1)               # [B]
        correct += int((pred == y).sum().item())
        total += int(y.numel())

    return correct / max(total, 1)

# =========================
# TRAIN
# =========================
def train_one_epoch(model, arcface, loader, optimizer, criterion, device, scaler=None):
    model.train()
    arcface.train()
    loss_sum = 0.0

    for x, y in tqdm(loader, desc="Training", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                feats = model.forward_features(x)
                emb = model.forward_head(feats, pre_logits=True)
                emb = F.normalize(emb, dim=1)
                logits = arcface(emb, y)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            if GRAD_CLIP_NORM is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(arcface.parameters()),
                    GRAD_CLIP_NORM
                )
            scaler.step(optimizer)
            scaler.update()
        else:
            feats = model.forward_features(x)
            emb = model.forward_head(feats, pre_logits=True)
            emb = F.normalize(emb, dim=1)
            logits = arcface(emb, y)
            loss = criterion(logits, y)

            loss.backward()
            if GRAD_CLIP_NORM is not None:
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(arcface.parameters()),
                    GRAD_CLIP_NORM
                )
            optimizer.step()

        loss_sum += float(loss.item())

    return loss_sum / max(len(loader), 1)

# =========================
# MAIN
# =========================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Saving to: ", SAVE_PATH)
    # Datasets
    train_ds = datasets.ImageFolder(Path(DATA_DIR) / "Training", transform=train_transform)
    val_ds   = datasets.ImageFolder(Path(DATA_DIR) / "Validation", transform=val_transform)

    # For prototype building (no aug)
    proto_ds = datasets.ImageFolder(Path(DATA_DIR) / "Training", transform=proto_transform)

    num_classes = len(train_ds.class_to_idx)
    assert num_classes > 1, "Need at least 2 classes."

    # Save mapping
    idx_to_class = {v: k for k, v in train_ds.class_to_idx.items()}
    with open(LABEL_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(idx_to_class, f, indent=2, ensure_ascii=False)

    # DataLoaders
    # On Windows, persistent_workers can sometimes be problematic; keep it True only when workers > 0.
    persistent = (NUM_WORKERS > 0)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"),
        persistent_workers=persistent
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"),
        persistent_workers=persistent
    )
    proto_loader = DataLoader(
        proto_ds, batch_size=64, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"),
        persistent_workers=persistent
    )

    # Backbone WITHOUT classifier
    model = timm.create_model(
        "convnextv2_base.fcmae_ft_in22k_in1k",
        pretrained=True,
        num_classes=0
    ).to(device)

    emb_dim = model.num_features
    arcface = ArcMarginProduct(
        in_features=emb_dim,
        out_features=num_classes,
        s=45.0,        # Increased for sharper decision boundaries (nighttime data)
        m=0.3          # Reduced margin for flexibility with low-quality images
    ).to(device)

    # Optional warmup freeze: train head first
    if WARMUP_FREEZE_EPOCHS > 0:
        for p in model.parameters():
            p.requires_grad = False

    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if p.requires_grad], "lr": 1e-4},
        {"params": arcface.parameters(), "lr": 1e-3},
    ], weight_decay=0.01)

    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    criterion = nn.CrossEntropyLoss()

    scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and device.type == "cuda"))

    best_val = 0.0
    best_epoch = -1

    early_stop = EarlyStopping(
        patience=EARLY_STOP_PATIENCE,
        min_delta=EARLY_STOP_MIN_DELTA
        )

    # Initial prototypes (optional)
    print("\nBuilding initial prototypes...")
    protos, counts = build_class_prototypes_from_loader(model, proto_loader, num_classes, device)
    torch.save({
        "prototypes": protos.detach().cpu(),
        "proto_labels": [idx_to_class[i] for i in range(num_classes)],
        "embedding_dim": emb_dim,
        "counts": counts,
        "epoch": -1
    }, LAST_PROTOS_PATH)

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")

        # Unfreeze backbone after warmup
        if epoch == WARMUP_FREEZE_EPOCHS and WARMUP_FREEZE_EPOCHS > 0:
            for p in model.parameters():
                p.requires_grad = True
            # Rebuild optimizer so backbone params are included
            optimizer = torch.optim.AdamW([
                {"params": model.parameters(), "lr": 1e-4},
                {"params": arcface.parameters(), "lr": 1e-3},
            ], weight_decay=0.01)
            scheduler = CosineAnnealingLR(optimizer, T_max=(NUM_EPOCHS - epoch))

        loss = train_one_epoch(model, arcface, train_loader, optimizer, criterion, device, scaler)

        # Rebuild prototypes periodically (so metric matches current model)
        if (epoch + 1) % BUILD_PROTOS_EVERY == 0:
            protos, counts = build_class_prototypes_from_loader(model, proto_loader, num_classes, device)
            
            # Warn about low-count classes (unreliable prototypes)
            MIN_SAMPLES_PER_CLASS = 5
            low_count_classes = [idx_to_class[i] for i in range(num_classes) if counts[i] < MIN_SAMPLES_PER_CLASS]
            if low_count_classes:
                print(f"  WARNING: Classes with <{MIN_SAMPLES_PER_CLASS} samples: {low_count_classes}")
            
            torch.save({
                "prototypes": protos.detach().cpu(),
                "proto_labels": [idx_to_class[i] for i in range(num_classes)],
                "embedding_dim": emb_dim,
                "counts": counts,
                "epoch": epoch + 1
            }, LAST_PROTOS_PATH)

        # Evaluate proper validation accuracy (prototype-based)
        if (epoch + 1) % EVAL_EVERY == 0:
            val_acc = prototype_accuracy(model, val_loader, protos, device)
        else:
            val_acc = float("nan")

        print(f"Loss: {loss:.6f} | Val Proto Acc: {val_acc:.4f}")

        # Save best backbone based on real metric
        if val_acc == val_acc and val_acc >= best_val:  # val_acc == val_acc filters NaN
            best_val = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            torch.save({
                "prototypes": protos.detach().cpu(),
                "proto_labels": [idx_to_class[i] for i in range(num_classes)],
                "embedding_dim": emb_dim,
                "counts": counts,
                "epoch": best_epoch
            }, BEST_PROTOS_PATH)

        torch.save(model.state_dict(), LAST_MODEL_SAVE_PATH)
        scheduler.step()

        # ---- EARLY STOPPING ----
        if (epoch + 1) % EVAL_EVERY == 0:
            if early_stop.step(val_acc):
                print(
                    f"Early stopping triggered at epoch {epoch+1}. "
                    f"Best Val Proto Acc={early_stop.best:.4f}."
                )
                break

    print(f"\nBest Val Proto Acc: {best_val:.4f} at epoch {best_epoch}")

    # Ensure final prototypes saved (already in LAST_PROTOS_PATH). Also write a canonical "prototypes.pt"
    # using the best backbone if available.
    if MODEL_SAVE_PATH.exists():
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
        protos, counts = build_class_prototypes_from_loader(model, proto_loader, num_classes, device)
        torch.save({
            "prototypes": protos.detach().cpu(),
            "proto_labels": [idx_to_class[i] for i in range(num_classes)],
            "embedding_dim": emb_dim,
            "counts": counts,
            "epoch": best_epoch
        }, SAVE_PATH / "prototypes.pt")

    print("Done.")

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
