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

        print(f"图像尺寸: {width} x {height}")

        channels = header['channels'].keys()
        print(f"可用通道: {list(channels)}")

        FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)

        # 找深度通道
        depth_channel_names = ['Z', 'depth', 'Depth', 'DEPTH', 'Y']
        depth_channel = None

        for name in depth_channel_names:
            if name in channels:
                depth_channel = name
                break

        print(f"使用通道: {depth_channel}")

        # 读取深度数据
        depth_str = exr_file.channel(depth_channel, FLOAT)
        depth_array = np.frombuffer(depth_str, dtype=np.float32)
        depth_array = depth_array.reshape((height, width))

        # 详细分析
        print("\n=== 深度图分析 ===")
        print(f"数据类型: {depth_array.dtype}")
        print(f"最小深度: {np.min(depth_array):.6f}")
        print(f"最大深度: {np.max(depth_array):.6f}")

        # 处理无穷大值
        finite_mask = np.isfinite(depth_array)
        finite_depths = depth_array[finite_mask]

        print(f"有限深度值数量: {len(finite_depths)} / {depth_array.size}")
        print(f"无穷大值数量: {np.sum(~finite_mask)}")

        if len(finite_depths) > 0:
            print(f"有限深度范围: {np.min(finite_depths):.6f} 到 {np.max(finite_depths):.6f}")
            print(f"有限深度均值: {np.mean(finite_depths):.6f}")
            print(f"有限深度中位数: {np.median(finite_depths):.6f}")

        return depth_array, finite_mask

    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None, None

def normalize_depth(depth_array, finite_mask, method='clip'):
    """
    归一化深度图到 0-1 范围

    method: 'clip' - 裁剪到最大有限值
            'percentile' - 使用百分位数
    """
    if method == 'clip':
        # 将无穷大值替换为最大有限值
        finite_depths = depth_array[finite_mask]
        if len(finite_depths) == 0:
            return depth_array

        max_depth = np.max(finite_depths)
        min_depth = np.min(finite_depths)

        # 创建副本并处理无穷大
        normalized = depth_array.copy()
        normalized[~finite_mask] = max_depth

        # 归一化到 0-1
        normalized = (normalized - min_depth) / (max_depth - min_depth)

    elif method == 'percentile':
        # 使用95%分位数作为最大值，避免极值影响
        finite_depths = depth_array[finite_mask]
        if len(finite_depths) == 0:
            return depth_array

        min_depth = np.percentile(finite_depths, 5)
        max_depth = np.percentile(finite_depths, 95)

        normalized = depth_array.copy()
        normalized[~finite_mask] = max_depth

        # 裁剪并归一化
        normalized = np.clip(normalized, min_depth, max_depth)
        normalized = (normalized - min_depth) / (max_depth - min_depth)

    return normalized

def save_normalized_depth(depth_array, output_path):
    """保存归一化后的深度图"""
    try:
        from PIL import Image
        # 转换为8位图像保存
        depth_8bit = (depth_array * 255).astype(np.uint8)
        img = Image.fromarray(depth_8bit, mode='L')  # 'L' 表示灰度图
        img.save(output_path)
        print(f"归一化深度图已保存到: {output_path}")
    except ImportError:
        print("需要安装 pillow 来保存图像")
        # 至少保存为 numpy 文件
        np.save(output_path.replace('.png', '.npy'), depth_array)
        print(f"深度数据已保存为 numpy 格式: {output_path.replace('.png', '.npy')}")

def main():
    if len(sys.argv) != 2:
        print("用法: python parse_depth.py <depth.exr文件路径>")
        return

    file_path = sys.argv[1]
    print(f"正在解析: {file_path}")

    depth_array, finite_mask = analyze_depth_exr(file_path)

    if depth_array is not None:
        print("\n=== 归一化选项 ===")

        # 方法1：裁剪归一化
        normalized_clip = normalize_depth(depth_array, finite_mask, 'clip')
        print(f"裁剪归一化后范围: {np.min(normalized_clip):.6f} 到 {np.max(normalized_clip):.6f}")

        # 方法2：百分位数归一化
        normalized_percentile = normalize_depth(depth_array, finite_mask, 'percentile')
        print(f"百分位数归一化后范围: {np.min(normalized_percentile):.6f} 到 {np.max(normalized_percentile):.6f}")

        # 保存归一化结果
        base_name = file_path.replace('.exr', '')
        save_normalized_depth(normalized_clip, f"{base_name}_normalized_clip.png")
        save_normalized_depth(normalized_percentile, f"{base_name}_normalized_percentile.png")

if __name__ == "__main__":
    main()