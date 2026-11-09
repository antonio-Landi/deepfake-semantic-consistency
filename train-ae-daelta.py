import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import lpips
import torch_dct
from pytorch_msssim import ssim

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
IMG_DIR          = '/path/to/training/images'
SAVE_PATH         = '/path/to/output/daelta.pth'
METRICS_SAVE_PATH = '/path/to/output/daelta-metrics.json'
CHECKPOINT_DIR    = '/path/to/output/daelta-checkpoints'

GPU_IDS    = [0]
DEVICE     = torch.device(f'cuda:{GPU_IDS[0]}' if torch.cuda.is_available() else 'cpu')

IMG_SIZE   = 256
DCT_SIZE   = 256
LATENT_DIM = 256
BATCH_SIZE = 32
EPOCHS     = 15
LR         = 1e-4
PATIENCE   = 7

# Loss weights
W_SSIM  = 0.30
W_LPIPS = 0.60
W_MSE   = 0.10


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class RecursiveImageDataset(Dataset):
    VALID_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')

    def __init__(self, folder, transform=None):
        if not os.path.isdir(folder):
            raise FileNotFoundError(f'Dataset folder not found: {folder}')

        self.paths = self._collect_images_recursive(folder)
        if not self.paths:
            raise RuntimeError(f'No valid images found in: {folder}')

        self.transform = transform
        self.folder = folder

    @classmethod
    def _collect_images_recursive(cls, folder):
        paths = []
        for root, dirs, files in os.walk(folder):
            dirs.sort()
            for name in sorted(files):
                if name.lower().endswith(cls.VALID_EXTENSIONS):
                    paths.append(os.path.join(root, name))
        return paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        with Image.open(self.paths[idx]) as img:
            img = img.convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img


def validate_config_paths():
    print('[CONFIG] Configured paths:')
    print(f'  DATASET:           {IMG_DIR}')
    print(f'  SAVE_PATH:         {SAVE_PATH}')
    print(f'  METRICS_SAVE_PATH: {METRICS_SAVE_PATH}')
    print(f'  CHECKPOINT_DIR:    {CHECKPOINT_DIR}')

    if not os.path.isdir(IMG_DIR):
        raise FileNotFoundError(f'Dataset folder not found: {IMG_DIR}')

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(METRICS_SAVE_PATH), exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def get_model_state_cpu(model):
    core_model = model.module if isinstance(model, nn.DataParallel) else model
    return {k: v.detach().cpu().clone() for k, v in core_model.state_dict().items()}


def build_config(dataset_count):
    return {
        'img_size':         IMG_SIZE,
        'dct_size':         DCT_SIZE,
        'latent_dim':       LATENT_DIM,
        'fused_latent':      LATENT_DIM * 2,
        'batch_size':       BATCH_SIZE,
        'epochs':           EPOCHS,
        'lr':               LR,
        'patience':         PATIENCE,
        'gpu_ids':          GPU_IDS,
        'save_path':        SAVE_PATH,
        'metrics_save_path': METRICS_SAVE_PATH,
        'checkpoint_dir':   CHECKPOINT_DIR,
        'w_ssim':           W_SSIM,
        'w_lpips':          W_LPIPS,
        'w_mse':            W_MSE,
        'dataset_dir':      IMG_DIR,
        'dataset_count':    dataset_count,
        'dct_encoder':      'conv3 + self-attention (GroupNorm+MHA+residual) + AdaptiveAvgPool2d(4) + Linear',
        'pixel_encoder':    'conv3 + flatten + Linear',
        'decoder':          'fully connected layer + transposed convolutions',
        'model':            'DAELTA',
    }


def write_metrics(metrics_path, epoch_history, best_epoch, best_loss, stopped_early, config):
    metrics = {
        'epochs_trained': len(epoch_history),
        'early_stopped': stopped_early,
        'best_epoch': best_epoch,
        'best_loss': best_loss,
        'final_loss': epoch_history[-1]['loss'] if epoch_history else None,
        'loss_history': [entry['loss'] for entry in epoch_history],
        'epoch_history': epoch_history,
        'config': config,
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    return metrics


# ---------------------------------------------------------------------------
# DCT branch transform
# ---------------------------------------------------------------------------
def to_dct_branch(x: torch.Tensor) -> torch.Tensor:
    """RGB [0,1] --> DCT log-scaled over 3 channels (B, 3, 256, 256)."""
    x_255 = x * 255.0
    dct   = torch_dct.dct_2d(x_255, norm='ortho')
    return torch.log(torch.abs(dct) + 1e-6)


# ---------------------------------------------------------------------------
# Self-attention for the DCT branch
# ---------------------------------------------------------------------------
class DCTSelfAttention(nn.Module):

    def __init__(self, channels: int = 128, heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=heads,
            batch_first=True,
        )
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        res = x

        x_norm = self.norm(x)
        seq = x_norm.flatten(2).permute(0, 2, 1)
        out, _ = self.attn(seq, seq, seq)
        out = out.permute(0, 2, 1).view(B, C, H, W)
        out = self.proj(out)

        return out + res


# ---------------------------------------------------------------------------
# Pixel encoder
# ---------------------------------------------------------------------------
class PixelEncoder(nn.Module):

    def __init__(self, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,   32,  4, 2, 1), nn.ReLU(),
            nn.Conv2d(32,  64,  4, 2, 1), nn.ReLU(),
            nn.Conv2d(64,  128, 4, 2, 1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 32 * 32, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# DCT encoder
# ---------------------------------------------------------------------------
class DCTEncoder(nn.Module):

    def __init__(self, latent_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,   32,  4, 2, 1), nn.ReLU(),
            nn.Conv2d(32,  64,  4, 2, 1), nn.ReLU(),
            nn.Conv2d(64,  128, 4, 2, 1), nn.ReLU(),
        )
        self.attn = DCTSelfAttention(channels=128, heads=4)
        self.pool = nn.AdaptiveAvgPool2d(64)
        self.fc   = nn.Linear(128 * 64 * 64, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.conv(x)
        f = self.attn(f)
        f = self.pool(f)
        return self.fc(f.flatten(1))


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------
class Decoder(nn.Module):

    def __init__(self, latent_dim: int):
        super().__init__()
        self.fc = nn.Linear(latent_dim * 2, 256 * 8 * 8)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64,  4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64,  32,  4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32,  3,   4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(self.fc(z).view(-1, 256, 8, 8))


# ---------------------------------------------------------------------------
# Complete model
# ---------------------------------------------------------------------------
class DualBranchAutoencoder(nn.Module):

    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.pixel_encoder = PixelEncoder(latent_dim)
        self.dct_encoder   = DCTEncoder(latent_dim)
        self.decoder       = Decoder(latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_pixel = self.pixel_encoder(x)
        z_dct   = self.dct_encoder(to_dct_branch(x))
        z       = torch.cat([z_pixel, z_dct], dim=1)
        return self.decoder(z)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
class ReconstructionLoss(nn.Module):
    """
    L = W_SSIM  * (1 - SSIM)
      + W_LPIPS * LPIPS
      + W_MSE   * MSE pixel
    """
    def __init__(self):
        super().__init__()
        self.lpips_fn = lpips.LPIPS(net='alex')

    def to(self, device):
        self.lpips_fn = self.lpips_fn.to(device)
        return super().to(device)

    def forward(self, recon: torch.Tensor, target: torch.Tensor):
        mse_loss   = F.mse_loss(recon, target)
        ssim_loss  = 1.0 - ssim(recon, target, data_range=1.0, size_average=True)
        lpips_loss = self.lpips_fn(recon * 2.0 - 1.0, target * 2.0 - 1.0).mean()

        total = (W_SSIM  * ssim_loss
               + W_LPIPS * lpips_loss
               + W_MSE   * mse_loss)

        return total, ssim_loss.item(), lpips_loss.item(), mse_loss.item()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train():
    validate_config_paths()

    print(f'Device: {DEVICE}')
    if torch.cuda.is_available():
        for gid in GPU_IDS:
            print(f'  GPU {gid}: {torch.cuda.get_device_name(gid)}')
    print('Model: DAELTA')
    print(f'  PixelEncoder: conv 3 layer + flatten + Linear  --> latent={LATENT_DIM}')
    print(f'  DCTEncoder:   conv 3 layer + self-attn + pool(4x4) + Linear  --> latent={LATENT_DIM}')
    print(f'  Fused latent:  {LATENT_DIM * 2}')
    print(f'Loss: W_SSIM={W_SSIM}  W_LPIPS={W_LPIPS}  W_MSE={W_MSE}')

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    dataset = RecursiveImageDataset(IMG_DIR, transform=transform)
    loader  = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    print('[DATASET] Images found:')
    print(f'  {IMG_DIR}: {len(dataset)}')
    print(f'Total images: {len(dataset)}  |  Batch: {len(loader)}  |  Batch size: {BATCH_SIZE}')
    config = build_config(len(dataset))

    model = DualBranchAutoencoder(LATENT_DIM).to(DEVICE)
    if torch.cuda.is_available() and len(GPU_IDS) > 1:
        model = nn.DataParallel(model, device_ids=GPU_IDS)
        print(f'DataParallel enabled on GPUs: {GPU_IDS}')

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable parameters: {n_params:,}')

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    criterion = ReconstructionLoss().to(DEVICE)

    epoch_history    = []
    best_loss        = float('inf')
    best_epoch       = 0
    patience_counter = 0
    best_state       = None
    stopped_early    = False
    n_batches        = len(loader)
    print_every      = max(1, n_batches // 10)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss       = 0.0
        epoch_ssim_loss  = 0.0
        epoch_lpips_loss = 0.0
        epoch_mse_loss   = 0.0

        for i, batch in enumerate(loader):
            batch = batch.to(DEVICE)
            recon = model(batch)
            loss, ssim_val, lpips_val, mse_val = criterion(recon, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_ssim_loss += ssim_val
            epoch_lpips_loss += lpips_val
            epoch_mse_loss += mse_val

            if (i + 1) % print_every == 0:
                print(
                    f'  epoch {epoch:>3} | batch {i+1:>4}/{n_batches} | '
                    f'loss: {loss.item():.4f} | '
                    f'ssim_loss: {ssim_val:.4f} | '
                    f'lpips_loss: {lpips_val:.4f} | '
                    f'mse_loss: {mse_val:.4f}',
                    flush=True,
                )

        avg_loss       = epoch_loss / n_batches
        avg_ssim_loss  = epoch_ssim_loss / n_batches
        avg_lpips_loss = epoch_lpips_loss / n_batches
        avg_mse_loss   = epoch_mse_loss / n_batches
        lr_before_step = optimizer.param_groups[0]['lr']
        scheduler.step(avg_loss)
        lr_after_step = optimizer.param_groups[0]['lr']

        state = get_model_state_cpu(model)
        improved = avg_loss < best_loss
        if avg_loss < best_loss:
            best_loss        = avg_loss
            best_epoch       = epoch
            patience_counter = 0
            best_state = state
            torch.save(best_state, SAVE_PATH)
            print(f'  --> Best DAELTA checkpoint updated  epoch {epoch}  loss: {best_loss:.6f}  [{SAVE_PATH}]', flush=True)
        else:
            patience_counter += 1

        epoch_checkpoint_path = os.path.join(CHECKPOINT_DIR, f'daelta-epoch-{epoch:03d}.pth')
        stopped_now = patience_counter >= PATIENCE
        if stopped_now:
            stopped_early = True

        epoch_metrics = {
            'epoch': epoch,
            'loss': avg_loss,
            'ssim_loss': avg_ssim_loss,
            'lpips_loss': avg_lpips_loss,
            'mse_loss': avg_mse_loss,
            'lr': lr_before_step,
            'next_lr': lr_after_step,
            'improved': improved,
            'best_epoch': best_epoch,
            'best_loss': best_loss,
            'patience_counter': patience_counter,
            'patience': PATIENCE,
            'checkpoint_path': epoch_checkpoint_path,
            'best_checkpoint_path': SAVE_PATH,
            'stopped_early': stopped_now,
        }

        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': state,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch_metrics': epoch_metrics,
                'config': config,
            },
            epoch_checkpoint_path,
        )
        epoch_history.append(epoch_metrics)
        write_metrics(METRICS_SAVE_PATH, epoch_history, best_epoch, best_loss, stopped_early, config)

        print(
            f'[Epoch {epoch}/{EPOCHS}] '
            f'loss={avg_loss:.6f} | '
            f'ssim_loss={avg_ssim_loss:.6f} | '
            f'lpips_loss={avg_lpips_loss:.6f} | '
            f'mse_loss={avg_mse_loss:.6f} | '
            f'lr={lr_before_step:.2e}->{lr_after_step:.2e} | '
            f'best_epoch={best_epoch} best_loss={best_loss:.6f} | '
            f'patience={patience_counter}/{PATIENCE}',
            flush=True,
        )
        print(f'  epoch checkpoint: {epoch_checkpoint_path}', flush=True)
        print(f'  metrics updated: {METRICS_SAVE_PATH}', flush=True)

        if stopped_now:
            print(f'\nEarly stopping at epoch {epoch}  best loss: {best_loss:.6f}')
            break

    if best_state is not None:
        core_model = model.module if isinstance(model, nn.DataParallel) else model
        core_model.load_state_dict(best_state)
        print(f'Best weights restored (epoch {best_epoch}  loss: {best_loss:.6f})')

    print(f'DAELTA model saved: {SAVE_PATH}')

    print(f'Metrics saved: {METRICS_SAVE_PATH}')


if __name__ == '__main__':
    train()
