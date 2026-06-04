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

# ============================================================================
# Debug: step through radar_polar_to_cartesian to find the bug
# ============================================================================
import cv2

azimuths = frame.azimuths
fft_data = frame.polar
resolution = frame.resolution
cart_resolution = 0.01
cart_pixel_width = 128

print(f"\n{'='*60}")
print("Step-by-step debugging of radar_polar_to_cartesian")
print(f"azimuths shape: {azimuths.shape}, dtype: {azimuths.dtype}")
print(f"fft_data shape: {fft_data.shape}, dtype: {fft_data.dtype}")
print(f"fft_data min/max: {fft_data.min():.4f} / {fft_data.max():.4f}")

# Step 1: cart_min_range calculation
if (cart_pixel_width % 2) == 0:
    cart_min_range = (cart_pixel_width / 2 - 0.5) * cart_resolution
else:
    cart_min_range = cart_pixel_width // 2 * cart_resolution
print(f"\ncart_min_range: {cart_min_range}")

# Step 2: coords
coords = np.linspace(-cart_min_range, cart_min_range, cart_pixel_width, dtype=np.float32)
Y, X = np.meshgrid(coords, -1 * coords)
print(f"Y shape: {Y.shape}, X shape: {X.shape}")

# Step 3: sample_range and sample_angle
sample_range = np.sqrt(Y * Y + X * X)
sample_angle = np.arctan2(Y, X)
sample_angle = sample_angle.astype(np.float32)
sample_angle += (sample_angle < 0).astype(np.float32) * 2.0 * np.pi
print(f"sample_range shape: {sample_range.shape}, min/max: {sample_range.min():.4f} / {sample_range.max():.4f}")
print(f"sample_angle shape: {sample_angle.shape}, min/max: {sample_angle.min():.4f} / {sample_angle.max():.4f}")

# Step 4: azimuth_step
azimuths_squeezed = azimuths.squeeze()
print(f"\nazimuths_squeezed shape: {azimuths_squeezed.shape}")
azimuth_step = (azimuths_squeezed[-1] - azimuths_squeezed[0]) / (azimuths_squeezed.shape[0] - 1)
print(f"azimuth_step: {azimuth_step}")

# Step 5: sample_u and sample_v
sample_u = (sample_range - resolution / 2) / resolution
sample_v = (sample_angle - azimuths_squeezed[0]) / azimuth_step
print(f"\nsample_u shape: {sample_u.shape}, min/max: {sample_u.min():.4f} / {sample_u.max():.4f}")
print(f"sample_v shape: {sample_v.shape}, min/max: {sample_v.min():.4f} / {sample_v.max():.4f}")

# Step 6: fix_wobble
M = azimuths_squeezed.shape[0]
if True:  # fix_wobble=True
    c3 = np.searchsorted(azimuths_squeezed, sample_angle.squeeze())
    c3[c3 == M] -= 1
    c2 = c3 - 1
    c2[c2 < 0] += 1
    a3 = azimuths_squeezed[c3]
    a2 = azimuths_squeezed[c2]
    diff = sample_angle.squeeze() - a3
    delta = diff * (diff < 0) * (c3 > 0) / (a3 - a2 + 1e-14)
    sample_v_fixed = (c3 + delta).astype(np.float32)
    print(f"\nAfter fix_wobble:")
    print(f"  sample_v_fixed shape: {sample_v_fixed.shape}, min/max: {sample_v_fixed.min():.4f} / {sample_v_fixed.max():.4f}")
    print(f"  c3 range: {c3.min()} - {c3.max()}")
    print(f"  c2 range: {c2.min()} - {c2.max()}")

# Step 7: interpolate_crossover
interpolate_crossover = True
if interpolate_crossover:
    fft_data_concat = np.concatenate((fft_data[-1:], fft_data, fft_data[:1]), 0)
    sample_v_final = sample_v_fixed + 1 if True else sample_v_fixed
    print(f"\nAfter interpolate_crossover:")
    print(f"  fft_data_concat shape: {fft_data_concat.shape}")
    print(f"  sample_v_final shape: {sample_v_final.shape}, min/max: {sample_v_final.min():.4f} / {sample_v_final.max():.4f}")
else:
    fft_data_concat = fft_data
    sample_v_final = sample_v_fixed if True else sample_v

# Step 8: warp
sample_u[sample_u < 0] = 0
polar_to_cart_warp = np.stack((sample_u, sample_v_final), -1)
print(f"\npolar_to_cart_warp shape: {polar_to_cart_warp.shape}")
print(f"polar_to_cart_warp dtype: {polar_to_cart_warp.dtype}")
print(f"polar_to_cart_warp min/max: {polar_to_cart_warp.min():.4f} / {polar_to_cart_warp.max():.4f}")

# Step 9: cv2.remap
print(f"\nfft_data_concat shape: {fft_data_concat.shape}, dtype: {fft_data_concat.dtype}")
print(f"fft_data_concat non-zero: {np.count_nonzero(fft_data_concat)}")

cart_result = cv2.remap(fft_data_concat, polar_to_cart_warp, None, cv2.INTER_LINEAR)
print(f"\nFinal result (original):")
print(f"  shape: {cart_result.shape}")
print(f"  min/max: {cart_result.min():.4f} / {cart_result.max():.4f}")
print(f"  non-zero pixels: {np.count_nonzero(cart_result)}")

# Test with transposed data
fft_data_t = fft_data_concat.T
polar_to_cart_warp_t = np.stack((sample_v_final, sample_u), -1)
cart_result_t = cv2.remap(fft_data_t, polar_to_cart_warp_t, None, cv2.INTER_LINEAR)
print(f"\nFinal result (transposed data):")
print(f"  shape: {cart_result_t.shape}")
print(f"  min/max: {cart_result_t.min():.4f} / {cart_result_t.max():.4f}")
print(f"  non-zero pixels: {np.count_nonzero(cart_result_t)}")

# Test with coarser resolution (0.0596 m/pixel to match radar resolution)
print(f"\n{'='*60}")
print("Testing with coarser resolution (0.0596 m/pixel)")
cart_coarse = frame.polar_to_cart(cart_resolution=0.0596, cart_pixel_width=128, in_place=False)
print(f"  shape: {cart_coarse.shape}, min/max: {cart_coarse.min():.4f} / {cart_coarse.max():.4f}")
print(f"  non-zero pixels: {np.count_nonzero(cart_coarse)}")

# Test with even coarser resolution
print(f"\nTesting with resolution 0.1 m/pixel")
cart_coarse2 = frame.polar_to_cart(cart_resolution=0.1, cart_pixel_width=128, in_place=False)
print(f"  shape: {cart_coarse2.shape}, min/max: {cart_coarse2.min():.4f} / {cart_coarse2.max():.4f}")
print(f"  non-zero pixels: {np.count_nonzero(cart_coarse2)}")
