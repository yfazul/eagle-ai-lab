"""
Test different OCR methods to see which one works with your APIs.
"""

import os
import base64
import requests
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image
import io

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Create a simple test image with text
def create_test_image():
    """Create a simple test image with text."""
    from PIL import Image, ImageDraw, ImageFont
    
    img = Image.new('RGB', (400, 100), color='white')
    draw = ImageDraw.Draw(img)
    
    # Use default font
    text = "Hello World - Test OCR"
    draw.text((10, 40), text, fill='black')
    
    # Convert to bytes
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return img_byte_arr.getvalue()

print("Creating test image...")
test_image_bytes = create_test_image()
base64_image = base64.b64encode(test_image_bytes).decode('utf-8')

print("\n" + "="*60)
print("TEST 1: OpenAI GPT-4 Vision OCR")
print("="*60)

try:
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Extract all text from this image. Return only the text, nothing else."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        max_tokens=500,
        temperature=0
    )
    
    text = response.choices[0].message.content
    print(f"✓ SUCCESS!")
    print(f"Extracted text: {text}")
    
except Exception as e:
    print(f"✗ FAILED: {e}")

print("\n" + "="*60)
print("TEST 2: DeepSeek Vision API (deepseek-chat)")
print("="*60)

try:
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Extract all text from this image. Return only the text."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "temperature": 0,
        "max_tokens": 500
    }
    
    response = requests.post(
        "https://api.deepseek.com/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=60
    )
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        print(f"✓ SUCCESS!")
        print(f"Extracted text: {text}")
    else:
        print(f"✗ FAILED!")
        print(f"Response: {response.text[:500]}")
    
except Exception as e:
    print(f"✗ FAILED: {e}")

print("\n" + "="*60)
print("TEST 3: DeepSeek with different model names")
print("="*60)

# Try different possible model names
model_names = [
    "deepseek-chat",
    "deepseek-vision",
    "deepseek-vl",
    "deepseek-reasoner",
]

for model_name in model_names:
    print(f"\nTrying model: {model_name}")
    
    try:
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, can you see images?"
                }
            ],
            "max_tokens": 100
        }
        
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            print(f"  ✓ Model {model_name} is available")
        else:
            print(f"  ✗ Model {model_name} failed: {response.status_code}")
            
    except Exception as e:
        print(f"  ✗ Error: {str(e)[:100]}")

print("\n" + "="*60)
print("RECOMMENDATION")
print("="*60)
print("Based on the tests above:")
print("1. If OpenAI works → Use OpenAI GPT-4 Vision for OCR")
print("2. If DeepSeek works → Use DeepSeek with the working model")
print("3. The updated code now uses OpenAI as fallback automatically")