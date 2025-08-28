#!/usr/bin/env python3
"""
MoGe API 客户端示例
演示如何使用 MoGe 模型的 HTTP API
"""

import requests
import sys
from pathlib import Path

def process_image(api_url, image_path, output_path, num_tokens=1024):
    """
    通过 HTTP API 处理图像
    
    Args:
        api_url: API 的基础 URL
        image_path: 输入图像的路径
        output_path: 输出图像的路径
        num_tokens: num_tokens 参数值
    """
    # 构建完整的 API URL
    process_url = f"{api_url.rstrip('/')}/process"
    
    try:
        # 准备文件和参数
        with open(image_path, 'rb') as f:
            files = {'image': (Path(image_path).name, f, 'image/jpeg')}
            data = {'num_tokens': num_tokens}
            
            # 发送 POST 请求
            print(f"正在处理图像: {image_path}")
            response = requests.post(process_url, files=files, data=data)
            
        # 检查响应
        if response.status_code == 200:
            # 保存结果
            with open(output_path, 'wb') as f:
                f.write(response.content)
            print(f"✅ 处理完成，结果已保存到: {output_path}")
            
            # 显示结果信息
            result_shape = response.headers.get('result-shape', 'unknown')
            print(f"📊 结果形状: {result_shape}")
        else:
            print(f"❌ 处理失败，状态码: {response.status_code}")
            print(f"错误信息: {response.text}")
            
    except FileNotFoundError:
        print(f"❌ 找不到图像文件: {image_path}")
    except Exception as e:
        print(f"❌ 处理过程中发生错误: {e}")

def main():
    if len(sys.argv) < 4:
        print("用法: python moge_client.py <API_URL> <输入图像路径> <输出图像路径> [num_tokens]")
        print("示例: python moge_client.py http://localhost:8000 ./assets/source.jpg ./output/result.png 1024")
        return
    
    api_url = sys.argv[1]
    image_path = sys.argv[2]
    output_path = sys.argv[3]
    num_tokens = int(sys.argv[4]) if len(sys.argv) > 4 else 1024
    
    process_image(api_url, image_path, output_path, num_tokens)

if __name__ == "__main__":
    main()