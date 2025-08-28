import onnxruntime as ort
import numpy as np
from PIL import Image
import io
import os
import tempfile
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import uvicorn

app = FastAPI(title="MoGe Model API", description="HTTP API for MoGe model inference")

# 全局变量存储模型会话和输入信息
model_session = None
model_inputs_info = None

def download_model():
    """从 Hugging Face 下载模型文件"""
    model_url = "https://huggingface.co/Ruicheng/moge-2-vitb-normal-onnx/resolve/main/model.onnx"
    model_path = "./models/moge_model.onnx"

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    if not os.path.exists(model_path):
        import requests
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

def preprocess_image(image_data):
    """预处理输入图像"""
    try:
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        original_size = image.size
        print(f"原始图像尺寸: {original_size}")

        # 调整图像大小
        target_size = (518, 518)
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        image_array = np.array(image).astype(np.float32) / 255.0
        image_array = np.transpose(image_array, (2, 0, 1))
        image_array = np.expand_dims(image_array, axis=0)

        print(f"预处理后形状: {image_array.shape}")
        return image_array, original_size
    except Exception as e:
        print(f"图像预处理失败: {e}")
        return None, None

def create_num_tokens_input(value):
    """创建 num_tokens 输入"""
    # 使用最简单有效的方法
    return np.array(value, dtype=np.int64).reshape(())

def run_inference(session, image_array, inputs_info, num_tokens_value=1024):
    """运行模型推理"""
    try:
        num_tokens_input = create_num_tokens_input(num_tokens_value)
        
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

        # 运行推理
        print(f"    推理中...")
        outputs = session.run(None, inputs)
        print(f"    ✅ 成功！")
        return outputs

    except Exception as e:
        error_msg = str(e)[:100]
        print(f"    ❌ 失败: {error_msg}")
        return None

def postprocess_output(outputs):
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
            elif normal_output.shape[1] == 3:  # (H, C, W) 格式
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

def save_result_to_temp_file(result):
    """将结果保存到临时文件并返回路径"""
    if result is None:
        return None

    try:
        # 创建临时文件
        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        temp_path = temp_file.name
        temp_file.close()

        print(f"💾 保存结果，形状: {result.shape}")

        # 如果是单通道（深度图或遮罩）
        if len(result.shape) == 2:
            # 归一化到 0-255
            normalized = ((result - result.min()) / (result.max() - result.min()) * 255).astype(np.uint8)
            Image.fromarray(normalized, mode='L').save(temp_path)
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
                Image.fromarray(result_8bit).save(temp_path)
                print(f"保存为RGB图像")

            elif result.shape[2] == 1:  # (H, W, 1) - 单通道但有额外维度
                result_2d = result[:, :, 0]
                normalized = ((result_2d - result_2d.min()) / (result_2d.max() - result_2d.min()) * 255).astype(np.uint8)
                Image.fromarray(normalized, mode='L').save(temp_path)
                print(f"保存为灰度图像（移除通道维度）")

            elif result.shape[0] == 3:  # (3, H, W) - 需要转置
                result_hwc = np.transpose(result, (1, 2, 0))
                if result_hwc.min() < 0:
                    normalized = (result_hwc + 1) / 2
                else:
                    normalized = np.clip(result_hwc, 0, 1)
                result_8bit = (normalized * 255).astype(np.uint8)
                Image.fromarray(result_8bit).save(temp_path)
                print(f"保存为RGB图像（CHW->HWC转换）")

            else:
                print(f"⚠️ 未知的3D形状: {result.shape}")
                # 尝试保存第一个通道
                if result.shape[0] < result.shape[2]:  # 假设第一个维度是通道
                    first_channel = result[0, :, :]
                else:
                    first_channel = result[:, :, 0]
                normalized = ((first_channel - first_channel.min()) / (first_channel.max() - first_channel.min()) * 255).astype(np.uint8)
                Image.fromarray(normalized, mode='L').save(temp_path)
                print(f"保存第一个通道为灰度图像")

        else:
            print(f"⚠️ 完全未知的形状: {result.shape}，只保存原始数据")

        # 检查文件是否真的创建了
        if os.path.exists(temp_path):
            file_size = os.path.getsize(temp_path)
            print(f"✅ 图像成功保存到临时文件: {temp_path} (大小: {file_size} 字节)")
            return temp_path
        else:
            print(f"❌ 图像文件未创建: {temp_path}")
            return None

    except Exception as e:
        print(f"❌ 保存失败: {e}")
        import traceback
        traceback.print_exc()
        return None

@app.on_event("startup")
async def load_model_on_startup():
    """在应用启动时加载模型"""
    global model_session, model_inputs_info
    
    print("正在加载模型...")
    model_path = download_model()
    if model_path is None:
        print("模型下载失败")
        return
    
    model_session, model_inputs_info = load_model(model_path)
    if model_session is None:
        print("模型加载失败")
        return
    
    print("模型加载完成")

@app.get("/")
async def root():
    """根路径，返回 API 信息"""
    return {
        "message": "MoGe Model API",
        "description": "HTTP API for MoGe model inference",
        "endpoints": {
            "POST /process": "Process an image and return the result",
            "GET /": "This information page"
        }
    }

@app.post("/process")
async def process_image(
    image: UploadFile = File(...),
    num_tokens: Optional[int] = 1024
):
    """处理上传的图像并返回结果"""
    global model_session, model_inputs_info
    
    if model_session is None:
        raise HTTPException(status_code=500, detail="模型未加载")
    
    try:
        # 读取上传的图像
        image_data = await image.read()
        
        # 预处理图像
        image_array, original_size = preprocess_image(image_data)
        if image_array is None:
            raise HTTPException(status_code=400, detail="图像处理失败")
        
        # 运行推理
        outputs = run_inference(model_session, image_array, model_inputs_info, num_tokens)
        if outputs is None:
            raise HTTPException(status_code=500, detail="模型推理失败")
        
        # 后处理输出
        result = postprocess_output(outputs)
        if result is None:
            raise HTTPException(status_code=500, detail="结果处理失败")
        
        # 保存结果到临时文件
        temp_path = save_result_to_temp_file(result)
        if temp_path is None:
            raise HTTPException(status_code=500, detail="结果保存失败")
        
        # 返回文件响应
        return FileResponse(
            temp_path,
            media_type="image/png",
            filename="result.png",
            headers={"result-shape": str(result.shape)}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理图像时发生错误: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)