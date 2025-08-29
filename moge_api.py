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

# Global variables to store model session and input information
model_session = None
model_inputs_info = None

def download_model():
    """Download model file from Hugging Face"""
    model_url = "https://huggingface.co/Ruicheng/moge-2-vitb-normal-onnx/resolve/main/model.onnx"
    model_path = "./models/moge_model.onnx"

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    if not os.path.exists(model_path):
        import requests
        print("Downloading model...")
        response = requests.get(model_url)
        if response.status_code == 200:
            with open(model_path, 'wb') as f:
                f.write(response.content)
            print("Model download completed")
        else:
            print(f"Download failed, status code: {response.status_code}")
            return None
    else:
        print("Model file already exists")

    return model_path

def load_model(model_path):
    """Load ONNX model"""
    try:
        # Try different execution providers
        providers = ['CPUExecutionProvider']
        if ort.get_available_providers():
            print(f"Available execution providers: {ort.get_available_providers()}")

        session = ort.InferenceSession(model_path, providers=providers)
        print("Model loaded successfully")

        print(f"Input information:")
        inputs_info = {}
        for input_meta in session.get_inputs():
            print(f"  Name: {input_meta.name}")
            print(f"  Shape: {input_meta.shape}")
            print(f"  Type: {input_meta.type}")
            inputs_info[input_meta.name] = {
                'shape': input_meta.shape,
                'type': input_meta.type,
                'meta': input_meta
            }

        print(f"Output information:")
        for output_meta in session.get_outputs():
            print(f"  Name: {output_meta.name}")
            print(f"  Shape: {output_meta.shape}")
            print(f"  Type: {output_meta.type}")

        return session, inputs_info
    except Exception as e:
        print(f"Model loading failed: {e}")
        return None, None

def preprocess_image(image_data):
    """Preprocess input image"""
    try:
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        original_size = image.size
        print(f"Original image size: {original_size}")

        # Resize image
        target_size = (518, 518)
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        image_array = np.array(image).astype(np.float32) / 255.0
        image_array = np.transpose(image_array, (2, 0, 1))
        image_array = np.expand_dims(image_array, axis=0)

        print(f"Preprocessed shape: {image_array.shape}")
        return image_array, original_size
    except Exception as e:
        print(f"Image preprocessing failed: {e}")
        return None, None

def create_num_tokens_input(value):
    """Create num_tokens input"""
    # Use the simplest effective method
    return np.array(value, dtype=np.int64).reshape(())

def run_inference(session, image_array, inputs_info, num_tokens_value=1024):
    """Run model inference"""
    try:
        num_tokens_input = create_num_tokens_input(num_tokens_value)
        
        inputs = {
            'image': image_array,
            'num_tokens': num_tokens_input
        }

        # Display input information
        for name, data in inputs.items():
            if hasattr(data, 'shape'):
                print(f"    {name}: shape={data.shape}, type={type(data)}, dtype={getattr(data, 'dtype', 'N/A')}")
            else:
                print(f"    {name}: value={data}, type={type(data)}")

        # Run inference
        print(f"    Running inference...")
        outputs = session.run(None, inputs)
        print(f"    ✅ Success!")
        return outputs

    except Exception as e:
        error_msg = str(e)[:100]
        print(f"    ❌ Failed: {error_msg}")
        return None

def postprocess_output(outputs):
    """Post-process output results"""
    if outputs is None or len(outputs) == 0:
        return None

    print(f"\n🎯 Output information:")
    for i, output in enumerate(outputs):
        print(f"  Output {i}: shape {output.shape}, range {output.min():.6f} to {output.max():.6f}")

    # According to model output, normal is the second output
    if len(outputs) >= 2:
        normal_output = outputs[1]  # normal
        print(f"Using normal output (outputs[1]), shape: {normal_output.shape}")

        # Process different output formats
        if len(normal_output.shape) == 4:  # (1, H, W, C) or (1, C, H, W)
            normal_output = normal_output[0]  # Remove batch dimension: (H, W, C) or (C, H, W)

            # Check if transpose is needed
            if normal_output.shape[0] == 3:  # (C, H, W) format
                normal_output = np.transpose(normal_output, (1, 2, 0))  # CHW -> HWC
            elif normal_output.shape[1] == 3:  # (H, C, W) format
                normal_output = np.transpose(normal_output, (0, 2, 1))  # HCW -> HWC
            # If (H, W, C) format, no transpose needed

        print(f"Post-processed shape: {normal_output.shape}")
        return normal_output

    # Use the first output
    output = outputs[0]
    if len(output.shape) == 4:  # (1, H, W, C) or (1, C, H, W)
        output = output[0]
        if output.shape[0] == 3:  # (C, H, W)
            output = np.transpose(output, (1, 2, 0))  # CHW -> HWC
        elif output.shape[1] == 3:  # (H, C, W)
            output = np.transpose(output, (0, 2, 1))  # HCW -> HWC
    elif len(output.shape) == 3 and output.shape[0] == 1:  # (1, H, W)
        output = output[0]

    return output

def save_result_to_temp_file(result):
    """Save result to temporary file and return path"""
    if result is None:
        return None

    try:
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        temp_path = temp_file.name
        temp_file.close()

        print(f"💾 Saving result, shape: {result.shape}")

        # If single channel (depth map or mask)
        if len(result.shape) == 2:
            # Normalize to 0-255
            normalized = ((result - result.min()) / (result.max() - result.min()) * 255).astype(np.uint8)
            Image.fromarray(normalized, mode='L').save(temp_path)
            print(f"Saved as grayscale image")

        # If three channels, ensure correct HWC format
        elif len(result.shape) == 3:
            if result.shape[2] == 3:  # (H, W, 3) - correct format
                # Normal map is usually in [-1, 1] range
                if result.min() < 0:
                    normalized = (result + 1) / 2  # [-1,1] -> [0,1]
                    print(f"Normal map range conversion: [{result.min():.3f}, {result.max():.3f}] -> [0, 1]")
                else:
                    normalized = np.clip(result, 0, 1)  # Ensure in [0,1] range

                result_8bit = (normalized * 255).astype(np.uint8)
                Image.fromarray(result_8bit).save(temp_path)
                print(f"Saved as RGB image")

            elif result.shape[2] == 1:  # (H, W, 1) - single channel but with extra dimension
                result_2d = result[:, :, 0]
                normalized = ((result_2d - result_2d.min()) / (result_2d.max() - result_2d.min()) * 255).astype(np.uint8)
                Image.fromarray(normalized, mode='L').save(temp_path)
                print(f"Saved as grayscale image (removed channel dimension)")

            elif result.shape[0] == 3:  # (3, H, W) - need to transpose
                result_hwc = np.transpose(result, (1, 2, 0))
                if result_hwc.min() < 0:
                    normalized = (result_hwc + 1) / 2
                else:
                    normalized = np.clip(result_hwc, 0, 1)
                result_8bit = (normalized * 255).astype(np.uint8)
                Image.fromarray(result_8bit).save(temp_path)
                print(f"Saved as RGB image (CHW->HWC conversion)")

            else:
                print(f"⚠️ Unknown 3D shape: {result.shape}")
                # Try to save the first channel
                if result.shape[0] < result.shape[2]:  # Assume first dimension is channel
                    first_channel = result[0, :, :]
                else:
                    first_channel = result[:, :, 0]
                normalized = ((first_channel - first_channel.min()) / (first_channel.max() - first_channel.min()) * 255).astype(np.uint8)
                Image.fromarray(normalized, mode='L').save(temp_path)
                print(f"Saved first channel as grayscale image")

        else:
            print(f"⚠️ Completely unknown shape: {result.shape}, only saving raw data")

        # Check if file was actually created
        if os.path.exists(temp_path):
            file_size = os.path.getsize(temp_path)
            print(f"✅ Image successfully saved to temporary file: {temp_path} (size: {file_size} bytes)")
            return temp_path
        else:
            print(f"❌ Image file not created: {temp_path}")
            return None

    except Exception as e:
        print(f"❌ Save failed: {e}")
        import traceback
        traceback.print_exc()
        return None

@app.on_event("startup")
async def load_model_on_startup():
    """Load model on application startup"""
    global model_session, model_inputs_info
    
    print("Loading model...")
    model_path = download_model()
    if model_path is None:
        print("Model download failed")
        return
    
    model_session, model_inputs_info = load_model(model_path)
    if model_session is None:
        print("Model loading failed")
        return
    
    print("Model loading completed")

@app.get("/")
async def root():
    """Root path, return API information"""
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
    """Process uploaded image and return result"""
    global model_session, model_inputs_info
    
    if model_session is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Read uploaded image
        image_data = await image.read()
        
        # Preprocess image
        image_array, original_size = preprocess_image(image_data)
        if image_array is None:
            raise HTTPException(status_code=400, detail="Image processing failed")
        
        # Run inference
        outputs = run_inference(model_session, image_array, model_inputs_info, num_tokens)
        if outputs is None:
            raise HTTPException(status_code=500, detail="Model inference failed")
        
        # Post-process output
        result = postprocess_output(outputs)
        if result is None:
            raise HTTPException(status_code=500, detail="Result processing failed")
        
        # Save result to temporary file
        temp_path = save_result_to_temp_file(result)
        if temp_path is None:
            raise HTTPException(status_code=500, detail="Result save failed")
        
        # Return file response
        return FileResponse(
            temp_path,
            media_type="image/png",
            filename="result.png",
            headers={"result-shape": str(result.shape)}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)