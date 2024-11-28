import os 
from PIL import Image
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    AutoProcessor,
    MusicgenForConditionalGeneration,
    LlamaForCausalLM,
    LlamaTokenizer,
)
import numpy as np
import gradio as gr
from scipy.io import wavfile
import torch
import json
from langchain_ollama import ChatOllama

# Dynamic hardware detection
def get_device():
    if torch.backends.mps.is_available():  # Check for Apple M1/M2
        return torch.device("mps")
    elif torch.cuda.is_available():  # Check for NVIDIA CUDA
        return torch.device("cuda")
    else:
        return torch.device("cpu")  # Default to CPU

# Load models
device = get_device()
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
music_processor = AutoProcessor.from_pretrained(r"facebook/musicgen-small")
music_model = MusicgenForConditionalGeneration.from_pretrained(r"facebook/musicgen-small").to(device)
llm = ChatOllama(model='llama3.2', format='json')
# Music generation function

def musicgen(img, text):
    raw_image = img.convert("RGB")

    # Generate image description using BLIP
    inputs = blip_processor(raw_image, text, return_tensors="pt").to(device)
    out = blip_model.generate(**inputs)
    blip_caption = blip_processor.decode(out[0], skip_special_tokens=True)

    # Optimize prompt using LLaMA
    input_text = f"Output a list of keywords that describe music that fits the image described by this caption '{blip_caption}'. Output in JSON format, using 'keywords' as a key."
    response = llm.invoke(input_text)

    # Initialize resp_json and optimized_caption
    resp_json = None
    optimized_caption = "A beautiful sunset on the beach, a car speeding by, jazz music"

    try:
        # Attempt to parse the response as JSON
        resp_json = json.loads(response.content)
        if isinstance(resp_json, dict) and "keywords" in resp_json:
            keywords = resp_json["keywords"]
            if isinstance(keywords, list):
                optimized_caption = ", ".join(keywords[:10])  # Limit to 10 keywords
    except json.JSONDecodeError:
        # If JSON parsing fails, log the error and use raw response content
        print("Failed to parse JSON. Using raw response content as fallback.")
        optimized_caption = response.content.strip()

    # Fallback to ensure optimized_caption is not empty
    if not optimized_caption.strip():
        optimized_caption = "A beautiful sunset on the beach, a car speeding by, jazz music"

    # Generate music using MusicGen
    music_inputs = music_processor(
        text=optimized_caption,
        padding=True,
        return_tensors="pt",
    ).to(device)
    audio_values = music_model.generate(**music_inputs, max_new_tokens=256)

    # Save the generated audio
    output_audio_path = os.path.join(os.path.dirname(__file__), "music/generated_audio.wav")
    os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)
    audio_data = audio_values[0, 0].cpu().numpy()  # Ensure output is on CPU
    wavfile.write(output_audio_path, music_model.config.audio_encoder.sampling_rate, audio_data)

    # Return the generated audio and optimized caption
    return output_audio_path, optimized_caption


# Memphis Style CSS with geometric shapes and background image
css = """
body {
    background-image: url('https://ice.frostsky.com/2024/11/28/670fcd3fb527f1330faf53b5ab45eea8.png');
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    margin: 0;
    padding: 0;
    height: 100%;
    width: 100%;
    display: flex;
    justify-content: center;
    align-items: center;
    color: white;  /* Optional: change text color for contrast */
}
.gradio-container {
    position: relative;
    z-index: 10;
    padding: 20px;
    background: rgba(0, 0, 0, 0.5); /* Optional: add semi-transparent background to enhance readability */
    border-radius: 10px;
}
"""

# Create Gradio interface
with gr.Blocks(gr.themes.Soft()) as demo:
    with gr.Column():
        with gr.Row():
            with gr.Column(scale=4):
                input_image = gr.Image(sources=['upload', 'webcam'], label="Upload pictures or use the camera to shoot", type='pil')
                # Use URL links for the example images
                gr.Examples(
                    [
                        'https://ice.frostsky.com/2024/11/28/f68692ac3cba0d305fa2132935c9890a.jpeg',  # 图片1的URL
                        'https://ice.frostsky.com/2024/11/28/5b4c6ff21ef07942755ffd4ae45699a5.jpeg',  # 图片2的URL
                        'https://ice.frostsky.com/2024/11/28/7b9f6fa3c897d198e1d62b1b868f10f2.png',  # 图片3的URL
                        'https://ice.frostsky.com/2024/11/28/b9075c9fd1ecd0e7decb729b4115233f.jpeg'   # 图片4的URL
                    ],
                    label="Sample Images", inputs=[input_image]
                )
            with gr.Column(scale=4):
                captioning = gr.Textbox(lines=2, label="Prompt")
                output_music = gr.Audio(type="numpy")
        button = gr.Button(value='Generate', variant="primary")
    
    # Button click event
    button.click(musicgen, [input_image, captioning], [output_music, captioning])

    # Apply custom CSS
    demo.css = css


demo.launch()