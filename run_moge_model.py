import onnxruntime as ort
import numpy as np
from PIL import Image
import requests
import os

def check_onnx_version():
    """检查 ONNX Runtime 版本"""
    print(f"ONNX Runtime 版本: {ort.__version__}")

def download_model():
    """从 Hugging Face 下载模型文件"""
    model_url = "https://huggingface.co/Ruicheng/moge-2-vitb-normal-onnx/resolve/main/model.onnx"
    model_path = "./models/moge_model.onnx"

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    if not os.path.exists(model_path):
        print("正在下载模型...")
        response = requests.get(model_url)
        if response.status_code == 200:
            with open(model_path, 'wb') as f:
                f.write(response.content)
            print("模型下载完成")
        else:
            print(f"下载失败，状态码: {response.status_code}")
            return None
    else:
        print("模型文件已存在")

    return model_path

def load_model(model_path):
    """加载 ONNX 模型"""
    try:
        # 尝试不同的执行提供者
        providers = ['CPUExecutionProvider']
        if ort.get_available_providers():
            print(f"可用的执行提供者: {ort.get_available_providers()}")

        session = ort.InferenceSession(model_path, providers=providers)
        print("模型加载成功")

        print(f"输入信息:")
        inputs_info = {}
        for input_meta in session.get_inputs():
            print(f"  名称: {input_meta.name}")
            print(f"  形状: {input_meta.shape}")
            print(f"  类型: {input_meta.type}")
            print(f"  类型详情: {str(input_meta.type)}")
            inputs_info[input_meta.name] = {
                'shape': input_meta.shape,
                'type': input_meta.type,
                'meta': input_meta
            }

        print(f"输出信息:")
        for output_meta in session.get_outputs():
            print(f"  名称: {output_meta.name}")
            print(f"  形状: {output_meta.shape}")
            print(f"  类型: {output_meta.type}")

        return session, inputs_info
    except Exception as e:
        print(f"模型加载失败: {e}")
        return None, None

def preprocess_image(image_path, target_size=(518, 518)):
    """预处理输入图像"""
    try:
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        print(f"原始图像尺寸: {original_size}")

        image = image.resize(target_size, Image.Resampling.LANCZOS)
        image_array = np.array(image).astype(np.float32) / 255.0
        image_array = np.transpose(image_array, (2, 0, 1))
        image_array = np.expand_dims(image_array, axis=0)

        print(f"预处理后形状: {image_array.shape}")
        return image_array, original_size
    except Exception as e:
        print(f"图像预处理失败: {e}")
        return None, None

def create_num_tokens_input(value, inputs_info):
    """创建不同格式的 num_tokens 输入"""

    methods = [
        # 方法1: 使用 numpy array 然后 reshape
        {
            'name': 'numpy array reshape',
            'func': lambda v: np.array(v, dtype=np.int64).reshape(())
        },
        # 方法2: 使用 numpy.asarray
        {
            'name': 'numpy asarray',
            'func': lambda v: np.asarray(v, dtype=np.int64)
        },
        # 方法3: 创建 0维数组
        {
            'name': 'zero-dim array',
            'func': lambda v: np.array(v, dtype=np.int64, ndmin=0)
        },
        # 方法4: 使用 numpy.atleast_0d
        {
            'name': 'atleast_0d',
            'func': lambda v: np.atleast_0d(np.int64(v))
        },
        # 方法5: 直接使用 numpy 标量构造
        {
            'name': 'numpy scalar constructor',
            'func': lambda v: np.int64(v)
        },
        # 方法6: 使用 OrtValue (如果可用)
        {
            'name': 'OrtValue',
            'func': lambda v: create_ort_value(v) if hasattr(ort, 'OrtValue') else None
        }
    ]

    for method in methods:
        try:
            result = method['func'](value)
            if result is not None:
                print(f"  {method['name']}: 类型={type(result)}, 形状={getattr(result, 'shape', 'N/A')}")
                yield method['name'], result
        except Exception as e:
            print(f"  {method['name']}: 失败 - {e}")

def create_ort_value(value):
    """尝试创建 OrtValue"""
    try:
        if hasattr(ort, 'OrtValue'):
            # 创建一个 numpy 数组然后转换为 OrtValue
            np_val = np.array(value, dtype=np.int64).reshape(())
            return ort.OrtValue.ortvalue_from_numpy(np_val)
        return None
    except:
        return None

def comprehensive_inference_test(session, image_array, inputs_info):
    """全面测试不同的输入组合"""

    check_onnx_version()

    # 可能的 num_tokens 值
    possible_values = [1024, 1025, 1089, 1036, 1156]  # 添加一些新值

    for value in possible_values:
        print(f"\n🔄 测试 num_tokens = {value}")

        for method_name, num_tokens_input in create_num_tokens_input(value, inputs_info):
            if num_tokens_input is None:
                continue

            try:
                print(f"\n  方法: {method_name}")

                inputs = {
                    'image': image_array,
                    'num_tokens': num_tokens_input
                }

                # 显示输入信息
                for name, data in inputs.items():
                    if hasattr(data, 'shape'):
                        print(f"    {name}: 形状={data.shape}, 类型={type(data)}, dtype={getattr(data, 'dtype', 'N/A')}")
                    else:
                        print(f"    {name}: 值={data}, 类型={type(data)}")

                # 尝试推理
                print(f"    推理中...")
                outputs = session.run(None, inputs)
                print(f"    ✅ 成功！")
                return outputs, value, method_name

            except Exception as e:
                error_msg = str(e)[:100]
                print(f"    ❌ 失败: {error_msg}")
                continue

    # 如果所有方法都失败，尝试分析模型的输入节点
    print(f"\n🔍 分析模型输入节点...")
    try:
        import onnx
        model = onnx.load(session._model_path if hasattr(session, '_model_path') else "./models/moge_model.onnx")

        for input_node in model.graph.input:
            if input_node.name == 'num_tokens':
                print(f"  ONNX 模型中的 num_tokens 定义:")
                print(f"    名称: {input_node.name}")
                print(f"    类型: {input_node.type}")
                if hasattr(input_node.type, 'tensor_type'):
                    print(f"    张量类型: {input_node.type.tensor_type}")
                    if hasattr(input_node.type.tensor_type, 'elem_type'):
                        print(f"    元素类型: {input_node.type.tensor_type.elem_type}")
    except Exception as e:
        print(f"  无法分析模型: {e}")

    return None, None, None

def postprocess_output(outputs, original_size):
    """后处理输出结果"""
    if outputs is None or len(outputs) == 0:
        return None

    print(f"\n🎯 输出信息:")
    for i, output in enumerate(outputs):
        print(f"  输出 {i}: 形状 {output.shape}, 范围 {output.min():.6f} 到 {output.max():.6f}")

    # 根据模型输出，normal 是第二个输出
    if len(outputs) >= 2:
        normal_output = outputs[1]  # normal
        print(f"使用法线输出 (outputs[1])，形状: {normal_output.shape}")

        # 处理不同的输出格式
        if len(normal_output.shape) == 4:  # (1, H, W, C) 或 (1, C, H, W)
            normal_output = normal_output[0]  # 移除 batch 维度: (H, W, C) 或 (C, H, W)

            # 检查是否需要转置
            if normal_output.shape[0] == 3:  # (C, H, W) 格式
                normal_output = np.transpose(normal_output, (1, 2, 0))  # CHW -> HWC
            elif normal_output.shape[1] == 3:  # (H, C, W) 格式 - 这是你遇到的情况
                normal_output = np.transpose(normal_output, (0, 2, 1))  # HCW -> HWC
            # 如果是 (H, W, C) 格式，不需要转置

        print(f"后处理后形状: {normal_output.shape}")
        return normal_output

    # 使用第一个输出
    output = outputs[0]
    if len(output.shape) == 4:  # (1, H, W, C) 或 (1, C, H, W)
        output = output[0]
        if output.shape[0] == 3:  # (C, H, W)
            output = np.transpose(output, (1, 2, 0))  # CHW -> HWC
        elif output.shape[1] == 3:  # (H, C, W)
            output = np.transpose(output, (0, 2, 1))  # HCW -> HWC
    elif len(output.shape) == 3 and output.shape[0] == 1:  # (1, H, W)
        output = output[0]

    return output

def save_result(result, output_path):
    """保存结果"""
    if result is None:
        return

    try:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

        print(f"💾 保存结果，形状: {result.shape}")

        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"创建输出目录: {output_dir}")

        # 如果是单通道（深度图或遮罩）
        if len(result.shape) == 2:
            # 归一化到 0-255
            normalized = ((result - result.min()) / (result.max() - result.min()) * 255).astype(np.uint8)
            Image.fromarray(normalized, mode='L').save(output_path)
            print(f"保存为灰度图像")

        # 如果是三通道，确保是正确的 HWC 格式
        elif len(result.shape) == 3:
            if result.shape[2] == 3:  # (H, W, 3) - 正确格式
                # 法线图通常在 [-1, 1] 范围内
                if result.min() < 0:
                    normalized = (result + 1) / 2  # [-1,1] -> [0,1]
                    print(f"法线图范围转换: [{result.min():.3f}, {result.max():.3f}] -> [0, 1]")
                else:
                    normalized = np.clip(result, 0, 1)  # 确保在 [0,1] 范围内

                result_8bit = (normalized * 255).astype(np.uint8)
                Image.fromarray(result_8bit).save(output_path)
                print(f"保存为RGB图像")

            elif result.shape[2] == 1:  # (H, W, 1) - 单通道但有额外维度
                result_2d = result[:, :, 0]
                normalized = ((result_2d - result_2d.min()) / (result_2d.max() - result_2d.min()) * 255).astype(np.uint8)
                Image.fromarray(normalized, mode='L').save(output_path)
                print(f"保存为灰度图像（移除通道维度）")

            elif result.shape[0] == 3:  # (3, H, W) - 需要转置
                result_hwc = np.transpose(result, (1, 2, 0))
                if result_hwc.min() < 0:
                    normalized = (result_hwc + 1) / 2
                else:
                    normalized = np.clip(result_hwc, 0, 1)
                result_8bit = (normalized * 255).astype(np.uint8)
                Image.fromarray(result_8bit).save(output_path)
                print(f"保存为RGB图像（CHW->HWC转换）")

            else:
                print(f"⚠️ 未知的3D形状: {result.shape}")
                # 尝试保存第一个通道
                if result.shape[0] < result.shape[2]:  # 假设第一个维度是通道
                    first_channel = result[0, :, :]
                else:
                    first_channel = result[:, :, 0]
                normalized = ((first_channel - first_channel.min()) / (first_channel.max() - first_channel.min()) * 255).astype(np.uint8)
                Image.fromarray(normalized, mode='L').save(output_path)
                print(f"保存第一个通道为灰度图像")

        else:
            print(f"⚠️ 完全未知的形状: {result.shape}，只保存原始数据")

        # 检查文件是否真的创建了
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"✅ 图像成功保存到: {output_path} (大小: {file_size} 字节)")
        else:
            print(f"❌ 图像文件未创建: {output_path}")

        # 保存原始数据
        npy_path = output_path.replace('.png', '.npy')
        np.save(npy_path, result)
        if os.path.exists(npy_path):
            print(f"✅ 原始数据保存到: {npy_path}")
        else:
            print(f"❌ 原始数据文件未创建: {npy_path}")

    except Exception as e:
        print(f"❌ 保存失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    import sys

    if len(sys.argv) != 2:
        print("用法: python run_moge_model.py <输入图像路径>")
        return

    input_image = sys.argv[1]

    model_path = download_model()
    if model_path is None:
        return

    session, inputs_info = load_model(model_path)
    if session is None:
        return

    image_array, original_size = preprocess_image(input_image)
    if image_array is None:
        return

    outputs, successful_value, successful_method = comprehensive_inference_test(session, image_array, inputs_info)
    if outputs is None:
        print("❌ 所有推理尝试都失败了")

        # 建议安装 onnx 来进一步分析
        print("\n💡 建议:")
        print("1. 尝试安装 onnx 包来分析模型: uv add onnx")
        print("2. 检查是否需要特定的 ONNX Runtime 版本")
        print("3. 查看原始模型的使用说明")
        return

    result = postprocess_output(outputs, original_size)

    base_name = os.path.splitext(os.path.basename(input_image))[0]
    output_path = f"./output/{base_name}_result.png"

    save_result(result, output_path)

    print(f"\n🎉 成功完成！")
    print(f"   使用方法: {successful_method}")
    print(f"   num_tokens 值: {successful_value}")

if __name__ == "__main__":
    main()