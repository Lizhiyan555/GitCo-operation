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
llm = ChatOllama(model='llama3.2')
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


# Gradio interface
with gr.Blocks(gr.themes.Soft()) as demo:
    with gr.Column():
        with gr.Row():
            with gr.Column(scale=4):
                input_image = gr.Image(sources=["upload"], label="Upload Image", type="pil")
                gr.Examples(
                    [r"/Users/shihuipeng/Documents/GitHub/GitCo-operation/ImagetoMusic_model_change/image2music/image_test/test1.jpg", r"/Users/shihuipeng/Documents/GitHub/GitCo-operation/ImagetoMusic_model_change/image2music/image_test/test2.jpg", r"/Users/shihuipeng/Documents/GitHub/GitCo-operation/ImagetoMusic_model_change/image2music/image_test/test3.png", r"/Users/shihuipeng/Documents/GitHub/GitCo-operation/ImagetoMusic_model_change/image2music/image_test/test4.jpg"],
                    label="Example Images",
                    inputs=[input_image],
                )
                button = gr.Button(value="Generate", variant="primary")
            with gr.Column(scale=4):
                captioning = gr.Textbox(lines=2, label="Prompt")
                output_music = gr.Audio(type="numpy")

    button.click(musicgen, [input_image, captioning], [output_music, captioning])

demo.launch()