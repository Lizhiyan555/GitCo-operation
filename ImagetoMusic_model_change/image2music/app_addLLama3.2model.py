# import streamlit as st # type: ignore
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoProcessor, MusicgenForConditionalGeneration, LlamaForCausalLM, LlamaTokenizer
import numpy as np
import gradio as gr
from scipy.io import wavfile

# Model
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cpu")
music_processor = AutoProcessor.from_pretrained(r"facebook/musicgen-small")
music_model = MusicgenForConditionalGeneration.from_pretrained(r"facebook/musicgen-small")
llama_tokenizer = LlamaTokenizer.from_pretrained("LLaMA-3.2-vision")
llama_model = LlamaForCausalLM.from_pretrained("LLaMA-3.2-vision")

# Music Generate Function
def musicgen(img, text):
    # Load the image and generate the description
    raw_image = img.convert('RGB')
    inputs = blip_processor(raw_image, text, return_tensors="pt").to("cpu")
    out = blip_model.generate(**inputs)
    blip_caption = blip_processor.decode(out[0], skip_special_tokens=True)

    # Added the LLaMA 3.2 optimization section
    # 提示优化的输入
    input_text = f"""
    Convert the following image description into a musical prompt for MusicGen. 
    - Highlight the emotional tone (e.g., peaceful, energetic).
    - Suggest a specific music style (e.g., jazz, orchestral, ambient).
    - Include relevant instruments or sound effects.

    Description: {blip_caption}
    """
    llama_inputs = llama_tokenizer(input_text, return_tensors="pt")
    llama_outputs = llama_model.generate(**llama_inputs, max_length=100)
    optimized_caption = llama_tokenizer.decode(llama_outputs[0], skip_special_tokens=True)

    # Generate music based on optimized descriptions
    music_inputs = music_processor(
        text=optimized_caption,
        padding=True,
        return_tensors="pt",
    )
    audio_values = music_model.generate(**music_inputs, max_new_tokens=256)

    # Converts the audio data type to numpy.int16
    audio_data = audio_values[0, 0].numpy()
    # Save audio file
    audio_filename = r".\music\generated_audio.wav"
    wavfile.write(audio_filename, music_model.config.audio_encoder.sampling_rate, audio_data)
    audio = wavfile.read(r".\music\generated_audio.wav")
    return audio, optimized_caption  # Returns optimized tips and audio

# Gradio UI
with gr.Blocks(gr.themes.Soft()) as demo:
    with gr.Column():
        with gr.Row():
            with gr.Column(scale=4):
                input_image = gr.Image(sources=['upload'], label="上传图片", type='pil')
                gr.Examples([r'.\image_test\test1.jpg', r'.\image_test\test2.jpg', r'.\image_test\test3.png', r'.\image_test\test4.jpg'],
                            label="参考范例", inputs=[input_image])
                button = gr.Button(value='生成', variant="primary")
            with gr.Column(scale=4):
                captioning = gr.Textbox(lines=2, label="文本提示")
                output_music = gr.Audio(type="numpy")
    
    button.click(musicgen, [input_image, captioning], [output_music, captioning])

demo.launch()
