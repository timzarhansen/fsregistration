import numpy as np
import os
from pyboreas import BoreasDataset

np.set_printoptions(precision=5, suppress=True)

root = "/home/tim-external/dataFolder/radar_boreas"
bd = BoreasDataset(root, split=None, verbose=True)
seq = bd.sequences[0]

print(f"\n{'='*60}")
print(f"Sequence: {seq.ID}")
print(f"Radar root: {seq.radar_root}")
print(f"Radar root exists: {os.path.isdir(seq.radar_root)}")
print(f"Number of radar frames: {len(seq.radar_frames)}")

# Check first radar frame path
frame = seq.get_radar(0)
print(f"\nFrame path: {frame.path}")
print(f"Frame file exists: {os.path.exists(frame.path)}")
print(f"Polar data: {frame.polar.shape if frame.polar is not None else 'None'}")
if frame.polar is not None:
    print(f"Polar data min/max: {frame.polar.min():.4f} / {frame.polar.max():.4f}")
    print(f"Polar data dtype: {frame.polar.dtype}")
    print(f"Polar data non-zero pixels: {np.count_nonzero(frame.polar)}")

# Convert to cartesian
cart = frame.polar_to_cart(cart_resolution=0.01, cart_pixel_width=128, in_place=False)
print(f"\nCartesian shape: {cart.shape}")
print(f"Cartesian min/max: {cart.min():.4f} / {cart.max():.4f}")
print(f"Cartesian dtype: {cart.dtype}")
print(f"Cartesian non-zero pixels: {np.count_nonzero(cart)}")

# Try a second frame
frame2 = seq.get_radar(1)
cart2 = frame2.polar_to_cart(cart_resolution=0.01, cart_pixel_width=128, in_place=False)
print(f"\nFrame 2 Cartesian min/max: {cart2.min():.4f} / {cart2.max():.4f}")

# Check azimuths
print(f"\nAzimuths shape: {frame.azimuths.shape if frame.azimuths is not None else 'None'}")
if frame.azimuths is not None:
    print(f"Azimuths min/max: {frame.azimuths.min():.4f} / {frame.azimuths.max():.4f}")

# Check resolution
print(f"Resolution: {frame.resolution}")
