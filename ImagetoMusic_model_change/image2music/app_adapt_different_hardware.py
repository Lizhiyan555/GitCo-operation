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
llama_tokenizer = LlamaTokenizer.from_pretrained("LLaMA-3.2-vision")
llama_model = LlamaForCausalLM.from_pretrained("LLaMA-3.2-vision").to(device)

# Music generation function
def musicgen(img, text):
    raw_image = img.convert("RGB")

    # Generate image description using BLIP
    inputs = blip_processor(raw_image, text, return_tensors="pt").to(device)
    out = blip_model.generate(**inputs)
    blip_caption = blip_processor.decode(out[0], skip_special_tokens=True)

    # Optimize prompt using LLaMA
    input_text = f"""
    Convert the following image description into a musical prompt for MusicGen. 
    - Highlight the emotional tone (e.g., peaceful, energetic).
    - Suggest a specific music style (e.g., jazz, orchestral, ambient).
    - Include relevant instruments or sound effects.

    Description: {blip_caption}
    """
    llama_inputs = llama_tokenizer(input_text, return_tensors="pt").to(device)
    llama_outputs = llama_model.generate(**llama_inputs, max_length=100)
    optimized_caption = llama_tokenizer.decode(llama_outputs[0], skip_special_tokens=True)

    # Generate music using MusicGen
    music_inputs = music_processor(
        text=optimized_caption,
        padding=True,
        return_tensors="pt",
    ).to(device)
    audio_values = music_model.generate(**music_inputs, max_new_tokens=256)

    # Convert audio to numpy array
    audio_data = audio_values[0, 0].cpu().numpy()  # Ensure output is on CPU
    audio_filename = r"./music/generated_audio.wav"
    wavfile.write(audio_filename, music_model.config.audio_encoder.sampling_rate, audio_data)
    audio = wavfile.read(r"./music/generated_audio.wav")
    return audio, optimized_caption

# Gradio interface
with gr.Blocks(gr.themes.Soft()) as demo:
    with gr.Column():
        with gr.Row():
            with gr.Column(scale=4):
                input_image = gr.Image(sources=["upload"], label="Upload Image", type="pil")
                gr.Examples(
                    [r"./image_test/test1.jpg", r"./image_test/test2.jpg", r"./image_test/test3.png", r"./image_test/test4.jpg"],
                    label="Example Images",
                    inputs=[input_image],
                )
                button = gr.Button(value="Generate", variant="primary")
            with gr.Column(scale=4):
                captioning = gr.Textbox(lines=2, label="Prompt")
                output_music = gr.Audio(type="numpy")

    button.click(musicgen, [input_image, captioning], [output_music, captioning])

demo.launch()