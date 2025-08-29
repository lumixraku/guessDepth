import OpenEXR
import Imath
import numpy as np
import sys

def analyze_depth_exr(file_path):
    try:
        exr_file = OpenEXR.InputFile(file_path)
        header = exr_file.header()
        dw = header['dataWindow']
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1

        print(f"Image size: {width} x {height}")

        channels = header['channels'].keys()
        print(f"Available channels: {list(channels)}")

        FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)

        # Find depth channel
        depth_channel_names = ['Z', 'depth', 'Depth', 'DEPTH', 'Y']
        depth_channel = None

        for name in depth_channel_names:
            if name in channels:
                depth_channel = name
                break

        print(f"Using channel: {depth_channel}")

        # Read depth data
        depth_str = exr_file.channel(depth_channel, FLOAT)
        depth_array = np.frombuffer(depth_str, dtype=np.float32)
        depth_array = depth_array.reshape((height, width))

        # Detailed analysis
        print("\n=== Depth Map Analysis ===")
        print(f"Data type: {depth_array.dtype}")
        print(f"Minimum depth: {np.min(depth_array):.6f}")
        print(f"Maximum depth: {np.max(depth_array):.6f}")

        # Handle infinite values
        finite_mask = np.isfinite(depth_array)
        finite_depths = depth_array[finite_mask]

        print(f"Finite depth values: {len(finite_depths)} / {depth_array.size}")
        print(f"Infinite values: {np.sum(~finite_mask)}")

        if len(finite_depths) > 0:
            print(f"Finite depth range: {np.min(finite_depths):.6f} to {np.max(finite_depths):.6f}")
            print(f"Finite depth mean: {np.mean(finite_depths):.6f}")
            print(f"Finite depth median: {np.median(finite_depths):.6f}")

        return depth_array, finite_mask

    except Exception as e:
        print(f"Error reading file: {e}")
        return None, None

def normalize_depth(depth_array, finite_mask, method='clip'):
    """
    Normalize depth map to 0-1 range

    method: 'clip' - clip to maximum finite value
            'percentile' - use percentiles
    """
    if method == 'clip':
        # Replace infinite values with maximum finite value
        finite_depths = depth_array[finite_mask]
        if len(finite_depths) == 0:
            return depth_array

        max_depth = np.max(finite_depths)
        min_depth = np.min(finite_depths)

        # Create copy and handle infinite values
        normalized = depth_array.copy()
        normalized[~finite_mask] = max_depth

        # Normalize to 0-1
        normalized = (normalized - min_depth) / (max_depth - min_depth)

    elif method == 'percentile':
        # Use 95th percentile as maximum value to avoid extreme values
        finite_depths = depth_array[finite_mask]
        if len(finite_depths) == 0:
            return depth_array

        min_depth = np.percentile(finite_depths, 5)
        max_depth = np.percentile(finite_depths, 95)

        normalized = depth_array.copy()
        normalized[~finite_mask] = max_depth

        # Clip and normalize
        normalized = np.clip(normalized, min_depth, max_depth)
        normalized = (normalized - min_depth) / (max_depth - min_depth)

    return normalized

def save_normalized_depth(depth_array, output_path):
    """Save normalized depth map"""
    try:
        from PIL import Image
        # Convert to 8-bit image for saving
        depth_8bit = (depth_array * 255).astype(np.uint8)
        img = Image.fromarray(depth_8bit, mode='L')  # 'L' represents grayscale
        img.save(output_path)
        print(f"Normalized depth map saved to: {output_path}")
    except ImportError:
        print("Pillow is required to save images")
        # At least save as numpy file
        np.save(output_path.replace('.png', '.npy'), depth_array)
        print(f"Depth data saved as numpy format: {output_path.replace('.png', '.npy')}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python parse_depth.py <depth.exr file path>")
        return

    file_path = sys.argv[1]
    print(f"Parsing: {file_path}")

    depth_array, finite_mask = analyze_depth_exr(file_path)

    if depth_array is not None:
        print("\n=== Normalization Options ===")

        # Method 1: Clipping normalization
        normalized_clip = normalize_depth(depth_array, finite_mask, 'clip')
        print(f"Clipping normalization range: {np.min(normalized_clip):.6f} to {np.max(normalized_clip):.6f}")

        # Method 2: Percentile normalization
        normalized_percentile = normalize_depth(depth_array, finite_mask, 'percentile')
        print(f"Percentile normalization range: {np.min(normalized_percentile):.6f} to {np.max(normalized_percentile):.6f}")

        # Save normalized results
        base_name = file_path.replace('.exr', '')
        save_normalized_depth(normalized_clip, f"{base_name}_normalized_clip.png")
        save_normalized_depth(normalized_percentile, f"{base_name}_normalized_percentile.png")

if __name__ == "__main__":
    main()