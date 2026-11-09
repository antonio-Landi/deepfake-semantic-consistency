import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import torch_dct

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WEIGHTS_PATH = '/path/to/model/daelta.pth'
FAKE_DIR     = '/path/to/input/images'
OUTPUT_DIR   = '/path/to/output/daelta-images'
N_IMAGES     = None

DEVICE     = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
IMG_SIZE   = 256
LATENT_DIM = 256


# ---------------------------------------------------------------------------
# DCT branch transform
# ---------------------------------------------------------------------------
def to_dct_branch(x: torch.Tensor) -> torch.Tensor:
    """RGB [0,1] --> DCT log-scaled over 3 channels (B, 3, 256, 256)."""
    x_255 = x * 255.0
    dct   = torch_dct.dct_2d(x_255, norm='ortho')
    return torch.log(torch.abs(dct) + 1e-6)


# ---------------------------------------------------------------------------
# Model architecture
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
        res    = x
        x_norm = self.norm(x)
        seq    = x_norm.flatten(2).permute(0, 2, 1)
        out, _ = self.attn(seq, seq, seq)
        out    = out.permute(0, 2, 1).view(B, C, H, W)
        return self.proj(out) + res


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
# Attack
# ---------------------------------------------------------------------------
def attack_image(model, img_path, transform_256, to_tensor):
    img_orig       = Image.open(img_path).convert('RGB')
    orig_w, orig_h = img_orig.size

    fake_orig = to_tensor(img_orig).unsqueeze(0).to(DEVICE)
    fake_256  = transform_256(img_orig).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        recon_256 = model(fake_256)

    delta_256  = recon_256 - fake_256
    delta_orig = F.interpolate(
        delta_256,
        size=(orig_h, orig_w),
        mode='bilinear',
        align_corners=False,
    )

    attacked     = (fake_orig + delta_orig).clamp(0, 1)
    attacked_pil = transforms.ToPILImage()(attacked.squeeze(0).cpu())
    return attacked_pil


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f'Device: {DEVICE}')
    if DEVICE.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(DEVICE.index)}')

    model = DualBranchAutoencoder(LATENT_DIM).to(DEVICE)
    
    state_dict = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)

    model.eval()
    print(f'DAELTA weights loaded from: {WEIGHTS_PATH}')

    transform_256 = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    to_tensor = transforms.ToTensor()

    valid_ext = ('.png', '.jpg', '.jpeg')
    all_files = sorted([
        f for f in os.listdir(FAKE_DIR)
        if f.lower().endswith(valid_ext)
    ])
    if N_IMAGES is not None:
        all_files = all_files[:N_IMAGES]

    print(f'Images to attack: {len(all_files)}')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f'Output: {OUTPUT_DIR}')

    for i, fname in enumerate(all_files):
        img_path = os.path.join(FAKE_DIR, fname)
        try:
            attacked_pil = attack_image(model, img_path, transform_256, to_tensor)
            attacked_pil.save(os.path.join(OUTPUT_DIR, fname))
            print(f'  [{i+1:>4}/{len(all_files)}] {fname} -> saved', flush=True)
        except Exception as e:
            print(f'  [{i+1:>4}/{len(all_files)}] {fname} -> ERROR: {e}', flush=True)

    print(f'\nCompleted. {len(all_files)} images saved to: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()