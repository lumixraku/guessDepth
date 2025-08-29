import onnxruntime as ort
import numpy as np
from PIL import Image
import requests
import os

def check_onnx_version():
    """Check ONNX Runtime version"""
    print(f"ONNX Runtime version: {ort.__version__}")

def download_model():
    """Download model file from Hugging Face"""
    model_url = "https://huggingface.co/Ruicheng/moge-2-vitb-normal-onnx/resolve/main/model.onnx"
    model_path = "./models/moge_model.onnx"

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    if not os.path.exists(model_path):
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
            print(f"  Type details: {str(input_meta.type)}")
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

def preprocess_image(image_path, target_size=(518, 518)):
    """Preprocess input image"""
    try:
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        print(f"Original image size: {original_size}")

        image = image.resize(target_size, Image.Resampling.LANCZOS)
        image_array = np.array(image).astype(np.float32) / 255.0
        image_array = np.transpose(image_array, (2, 0, 1))
        image_array = np.expand_dims(image_array, axis=0)

        print(f"Preprocessed shape: {image_array.shape}")
        return image_array, original_size
    except Exception as e:
        print(f"Image preprocessing failed: {e}")
        return None, None

def create_num_tokens_input(value, inputs_info):
    """Create different formats of num_tokens input"""

    methods = [
        # Method 1: Use numpy array then reshape
        {
            'name': 'numpy array reshape',
            'func': lambda v: np.array(v, dtype=np.int64).reshape(())
        },
        # Method 2: Use numpy.asarray
        {
            'name': 'numpy asarray',
            'func': lambda v: np.asarray(v, dtype=np.int64)
        },
        # Method 3: Create 0-dim array
        {
            'name': 'zero-dim array',
            'func': lambda v: np.array(v, dtype=np.int64, ndmin=0)
        },
        # Method 4: Use numpy.atleast_0d
        {
            'name': 'atleast_0d',
            'func': lambda v: np.atleast_0d(np.int64(v))
        },
        # Method 5: Use numpy scalar constructor
        {
            'name': 'numpy scalar constructor',
            'func': lambda v: np.int64(v)
        },
        # Method 6: Use OrtValue (if available)
        {
            'name': 'OrtValue',
            'func': lambda v: create_ort_value(v) if hasattr(ort, 'OrtValue') else None
        }
    ]

    for method in methods:
        try:
            result = method['func'](value)
            if result is not None:
                print(f"  {method['name']}: type={type(result)}, shape={getattr(result, 'shape', 'N/A')}")
                yield method['name'], result
        except Exception as e:
            print(f"  {method['name']}: failed - {e}")

def create_ort_value(value):
    """Try to create OrtValue"""
    try:
        if hasattr(ort, 'OrtValue'):
            # Create a numpy array then convert to OrtValue
            np_val = np.array(value, dtype=np.int64).reshape(())
            return ort.OrtValue.ortvalue_from_numpy(np_val)
        return None
    except:
        return None

def comprehensive_inference_test(session, image_array, inputs_info):
    """Comprehensive test of different input combinations"""

    check_onnx_version()

    # Possible num_tokens values
    possible_values = [1024, 1025, 1089, 1036, 1156]  # Add some new values

    for value in possible_values:
        print(f"\n🔄 Testing num_tokens = {value}")

        for method_name, num_tokens_input in create_num_tokens_input(value, inputs_info):
            if num_tokens_input is None:
                continue

            try:
                print(f"\n  Method: {method_name}")

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

                # Try inference
                print(f"    Running inference...")
                outputs = session.run(None, inputs)
                print(f"    ✅ Success!")
                return outputs, value, method_name

            except Exception as e:
                error_msg = str(e)[:100]
                print(f"    ❌ Failed: {error_msg}")
                continue

    # If all methods fail, try to analyze the model's input nodes
    print(f"\n🔍 Analyzing model input nodes...")
    try:
        import onnx
        model = onnx.load(session._model_path if hasattr(session, '_model_path') else "./models/moge_model.onnx")

        for input_node in model.graph.input:
            if input_node.name == 'num_tokens':
                print(f"  num_tokens definition in ONNX model:")
                print(f"    Name: {input_node.name}")
                print(f"    Type: {input_node.type}")
                if hasattr(input_node.type, 'tensor_type'):
                    print(f"    Tensor type: {input_node.type.tensor_type}")
                    if hasattr(input_node.type.tensor_type, 'elem_type'):
                        print(f"    Element type: {input_node.type.tensor_type.elem_type}")
    except Exception as e:
        print(f"  Unable to analyze model: {e}")

    return None, None, None

def postprocess_output(outputs, original_size):
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
            elif normal_output.shape[1] == 3:  # (H, C, W) format - this is what you encountered
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

def save_result(result, output_path):
    """Save result"""
    if result is None:
        return

    try:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

        print(f"💾 Saving result, shape: {result.shape}")

        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

        # If single channel (depth map or mask)
        if len(result.shape) == 2:
            # Normalize to 0-255
            normalized = ((result - result.min()) / (result.max() - result.min()) * 255).astype(np.uint8)
            Image.fromarray(normalized, mode='L').save(output_path)
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
                Image.fromarray(result_8bit).save(output_path)
                print(f"Saved as RGB image")

            elif result.shape[2] == 1:  # (H, W, 1) - single channel but with extra dimension
                result_2d = result[:, :, 0]
                normalized = ((result_2d - result_2d.min()) / (result_2d.max() - result_2d.min()) * 255).astype(np.uint8)
                Image.fromarray(normalized, mode='L').save(output_path)
                print(f"Saved as grayscale image (removed channel dimension)")

            elif result.shape[0] == 3:  # (3, H, W) - need to transpose
                result_hwc = np.transpose(result, (1, 2, 0))
                if result_hwc.min() < 0:
                    normalized = (result_hwc + 1) / 2
                else:
                    normalized = np.clip(result_hwc, 0, 1)
                result_8bit = (normalized * 255).astype(np.uint8)
                Image.fromarray(result_8bit).save(output_path)
                print(f"Saved as RGB image (CHW->HWC conversion)")

            else:
                print(f"⚠️ Unknown 3D shape: {result.shape}")
                # Try to save the first channel
                if result.shape[0] < result.shape[2]:  # Assume first dimension is channel
                    first_channel = result[0, :, :]
                else:
                    first_channel = result[:, :, 0]
                normalized = ((first_channel - first_channel.min()) / (first_channel.max() - first_channel.min()) * 255).astype(np.uint8)
                Image.fromarray(normalized, mode='L').save(output_path)
                print(f"Saved first channel as grayscale image")

        else:
            print(f"⚠️ Completely unknown shape: {result.shape}, only saving raw data")

        # Check if file was actually created
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"✅ Image successfully saved to: {output_path} (size: {file_size} bytes)")
        else:
            print(f"❌ Image file not created: {output_path}")

        # Save raw data
        npy_path = output_path.replace('.png', '.npy')
        np.save(npy_path, result)
        if os.path.exists(npy_path):
            print(f"✅ Raw data saved to: {npy_path}")
        else:
            print(f"❌ Raw data file not created: {npy_path}")

    except Exception as e:
        print(f"❌ Save failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    import sys

    if len(sys.argv) != 2:
        print("Usage: python run_moge_model.py <input_image_path>")
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
        print("❌ All inference attempts failed")

        # Suggest installing onnx for further analysis
        print("\n💡 Suggestions:")
        print("1. Try installing the onnx package to analyze the model: uv add onnx")
        print("2. Check if a specific ONNX Runtime version is needed")
        print("3. Check the original model's usage instructions")
        return

    result = postprocess_output(outputs, original_size)

    base_name = os.path.splitext(os.path.basename(input_image))[0]
    output_path = f"./output/{base_name}_result.png"

    save_result(result, output_path)

    print(f"\n🎉 Success completed!")
    print(f"   Method used: {successful_method}")
    print(f"   num_tokens value: {successful_value}")

if __name__ == "__main__":
    main()