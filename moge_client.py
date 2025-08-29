#!/usr/bin/env python3
"""
MoGe API Client Example
Demonstrates how to use the MoGe model's HTTP API
"""

import requests
import sys
from pathlib import Path

def process_image(api_url, image_path, output_path, num_tokens=1024):
    """
    Process image through HTTP API
    
    Args:
        api_url: Base URL of the API
        image_path: Path to the input image
        output_path: Path to the output image
        num_tokens: num_tokens parameter value
    """
    # Build the complete API URL
    process_url = f"{api_url.rstrip('/')}/process"
    
    try:
        # Prepare file and parameters
        with open(image_path, 'rb') as f:
            files = {'image': (Path(image_path).name, f, 'image/jpeg')}
            data = {'num_tokens': num_tokens}
            
            # Send POST request
            print(f"Processing image: {image_path}")
            response = requests.post(process_url, files=files, data=data)
            
        # Check response
        if response.status_code == 200:
            # Save result
            with open(output_path, 'wb') as f:
                f.write(response.content)
            print(f"✅ Processing completed, result saved to: {output_path}")
            
            # Display result information
            result_shape = response.headers.get('result-shape', 'unknown')
            print(f"📊 Result shape: {result_shape}")
        else:
            print(f"❌ Processing failed, status code: {response.status_code}")
            print(f"Error message: {response.text}")
            
    except FileNotFoundError:
        print(f"❌ Image file not found: {image_path}")
    except Exception as e:
        print(f"❌ Error occurred during processing: {e}")

def main():
    if len(sys.argv) < 4:
        print("Usage: python moge_client.py <API_URL> <input_image_path> <output_image_path> [num_tokens]")
        print("Example: python moge_client.py http://localhost:8000 ./assets/source.jpg ./output/result.png 1024")
        return
    
    api_url = sys.argv[1]
    image_path = sys.argv[2]
    output_path = sys.argv[3]
    num_tokens = int(sys.argv[4]) if len(sys.argv) > 4 else 1024
    
    process_image(api_url, image_path, output_path, num_tokens)

if __name__ == "__main__":
    main()