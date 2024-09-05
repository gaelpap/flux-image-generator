from flask import Flask, render_template, request, send_file, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import os
import fal_client
import requests
from io import BytesIO

app = Flask(__name__)

# Set up rate limiting
limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# Ensure API key is set as an environment variable
if 'FAL_KEY' not in os.environ:
    raise EnvironmentError("FAL_KEY environment variable is not set")

def generate_image(prompt, loras, enable_safety_checker):
    arguments = {
        "prompt": prompt,
        "image_size": "landscape_4_3",
        "num_inference_steps": 28,
        "guidance_scale": 3.5,
        "num_images": 1,
        "enable_safety_checker": enable_safety_checker,
        "output_format": "jpeg",
    }
    
    if loras:
        arguments["loras"] = loras

    try:
        handler = fal_client.submit("fal-ai/flux-lora", arguments=arguments)
        result = handler.get()
        image_url = result['images'][0]['url']
        response = requests.get(image_url)
        if response.status_code == 200:
            return BytesIO(response.content)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    return None

@app.route('/', methods=['GET', 'POST'])
@limiter.limit("10 per minute")
def index():
    if request.method == 'POST':
        prompt = request.form['prompt']
        loras = []
        for i in range(3):
            lora_path = request.form.get(f'lora_path_{i}')
            lora_scale = request.form.get(f'lora_scale_{i}')
            if lora_path and lora_scale:
                loras.append({"path": lora_path, "scale": float(lora_scale)})
        enable_safety = request.form.get('enable_safety') == 'on'
        
        image = generate_image(prompt, loras, enable_safety)
        if image:
            return send_file(image, mimetype='image/jpeg', as_attachment=True, download_name='generated_image.jpg')
        else:
            return jsonify({"error": "Failed to generate image"}), 500
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))