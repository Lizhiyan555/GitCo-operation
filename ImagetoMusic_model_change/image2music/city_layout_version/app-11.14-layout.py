from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoProcessor, MusicgenForConditionalGeneration
import numpy as np
import gradio as gr
from scipy.io import wavfile

# 音乐生成函数
def musicgen(img, text):
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cpu")

    music_processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    music_model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
    
    raw_image = img.convert('RGB')
    inputs = blip_processor(raw_image, text, return_tensors="pt").to("cpu")
    out = blip_model.generate(**inputs)
    caption = blip_processor.decode(out[0], skip_special_tokens=True)

    music_inputs = music_processor(
        text=caption,
        padding=True,
        return_tensors="pt",
    )
    audio_values = music_model.generate(**music_inputs, max_new_tokens=256)

    audio_data = audio_values[0,0].numpy()
    audio_filename = "./music/generated_audio.wav"
    wavfile.write(audio_filename, music_model.config.audio_encoder.sampling_rate, audio_data)
    audio = wavfile.read("./music/generated_audio.wav")
    return audio, caption

with gr.Blocks(css="custom_style.css") as demo:
    gr.Markdown(
        """
        <h1 style="text-align: center; color: #ff33ff;">Pls upload a photo to generate a music</h1>
        <p style="text-align: center; font-size: 18px; color: #333333;">
        upload a picture Pls <strong>生成</strong> push the button to generate a music!</p>
        """, 
        elem_id="header"
    )
    with gr.Row():
        with gr.Column(scale=4):
            # 修改后的输入图像组件
            input_image = gr.Image(type="pil", label="upload pic", elem_classes=["upload-image-area"])
            button = gr.Button(value="generate", elem_id="generate-button")
        with gr.Column(scale=4):
            captioning = gr.Textbox(lines=2, label="text hint", elem_classes=["textbox"])
            output_music = gr.Audio(type="numpy", elem_classes=["audio-player"])
    
    button.click(musicgen, [input_image, captioning], [output_music, captioning])

demo.launch()
