import random
from PIL import Image, ImageDraw
import os
import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
from diffusers import EulerAncestralDiscreteScheduler
import sys
import gc
import re
from dotenv import load_dotenv
import gradio as gr
import hashlib
from PIL.PngImagePlugin import PngInfo
from datetime import datetime as date_time
from compel import Compel, ReturnedEmbeddingsType

def calculate_sha256(filename, cut=10): # for everything except for LoRA
    hash_sha256 = hashlib.sha256()
    blksize = 1024 * 1024

    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(blksize), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()[:cut]

def addnet_hash_safetensors(b, cut=12): # for LoRA
    """kohya-ss hash for safetensors from https://github.com/kohya-ss/sd-scripts/blob/main/library/train_util.py"""
    hash_sha256 = hashlib.sha256()
    blksize = 1024 * 1024

    b.seek(0)
    header = b.read(8)
    n = int.from_bytes(header, "little")

    offset = n + 8
    b.seek(offset)
    for chunk in iter(lambda: b.read(blksize), b""):
        hash_sha256.update(chunk)

    return hash_sha256.hexdigest()[:cut]

device_ = "cuda"
torch_dtype_ = torch.float16

path = os.path.abspath("src")
sys.path.append(path)

env_path = os.path.join('env', 'manga_generator.env')
if os.path.exists(env_path):
    load_dotenv(env_path)

MAIN_DIR = os.getenv("main_dir", "./")
if MAIN_DIR[-1] != "/":
    MAIN_DIR = MAIN_DIR + "/"
MODEL_PATH = MAIN_DIR + os.getenv('MODEL_PATH', 'models/Stable-diffusion/')
CHECKPOINT = os.getenv('CHECKPOINT', 'ponyDiffusionV6XL_v6StartWithThisOne.safetensors')
VAE_PATH = MAIN_DIR + os.getenv('VAE_PATH', 'models/VAE/')
if VAE_PATH.startswith('/'):
    VAE_PATH = VAE_PATH[1:]
MODEL_FILE_VAE = os.getenv('MODEL_FILE_VAE', '')
LORA_PATH = MAIN_DIR + os.getenv('LORA_PATH', 'Lora/')
if LORA_PATH.startswith('/'):
    LORA_PATH = LORA_PATH[1:]
MODEL_FILE_LORA = os.getenv('MODEL_FILE_LORA', '').split(",")
LORA_W = os.getenv('LORA_W', '1').split(",")
NEGATIVE_PROMPT = os.getenv('NEGATIVE_PROMPT', 'blurry,blurry_image,lowres,low_resolution,low_picture_quality')
WIDTH = int(os.getenv('WIDTH', 826))
HEIGHT = int(os.getenv('HEIGHT', 1164))
MARGIN = int(os.getenv('MARGIN', 10))
MANGA_DIR = os.getenv('MANGA_DIR', 'manga')
if MANGA_DIR[-1] != "/":
    MANGA_DIR = MANGA_DIR + "/"
PROMPT_FILE = os.getenv('PROMPT_FILE', 'manga_generator.txt')
CFG = int(os.getenv('CFG', '7'))
STEPS = int(os.getenv('STEPS', '40'))
INPUT_SEED = int(os.getenv('INPUT_SEED', '-1'))
MANGA_SCENES_ORDER = os.getenv('MANGA_SCENES_ORDER', 'true').lower() == "true"

def count_file(directory_path_temp):
    unique_id_temp = 0
    existing_files = len([f for f in os.listdir(directory_path_temp) if f.endswith(".png") and os.path.isfile(os.path.join(directory_path_temp, f))])
    unique_id_temp = existing_files + 1
    return unique_id_temp

def read_prompts(prompt_input, input):
    if input == "file":
        with open(prompt_input, 'r') as file:
            lines = file.readlines()
    else:
        lines = prompt_input.splitlines()
    pages = []
    current_page = []
    for line in lines:
        if line.strip() == "":
            if current_page:
                pages.append(current_page)
                current_page = []
        else:
            if len(current_page) < 6:
                current_page.append(line.strip())
    if current_page:
        pages.append(current_page)
    return pages

def image_print_create(prompt, width_param, height_param, negative_prompt, cfg, steps, input_seed):

    while True:
        if input_seed > -1:
            seed = input_seed
        else:
            seed = random.randint(0, 9999999999)
        generator = torch.Generator(device=device_).manual_seed(seed)

        pipeline.safety_checker = None
        pipeline.requires_safety_checker = False

        width_ = width_param
        height_ = height_param
        resize_pixel_w = width_ % 8
        resize_pixel_h = height_ % 8

        if resize_pixel_w > 0:
            width_ = width_ - resize_pixel_w
        if resize_pixel_h > 0:
            height_ = height_ - resize_pixel_h

        guidance_scale = cfg
        num_inference_steps = steps

        compel = Compel(
        tokenizer=[pipeline.tokenizer, pipeline.tokenizer_2] ,
        text_encoder=[pipeline.text_encoder, pipeline.text_encoder_2],
        returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
        requires_pooled=[False, True],
        truncate_long_prompts=False
        )
        conditioning, pooled = compel.build_conditioning_tensor(prompt)
        negative_conditioning, negative_pooled = compel.build_conditioning_tensor(negative_prompt)
        [conditioning, negative_conditioning] = compel.pad_conditioning_tensors_to_same_length([conditioning, negative_conditioning])

        image = pipeline(prompt_embeds=conditioning, negative_prompt_embeds=negative_conditioning, pooled_prompt_embeds=pooled, negative_pooled_prompt_embeds=negative_pooled, generator=generator, width=width_, height=height_, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images[0]

        if resize_pixel_w > 0 or resize_pixel_h > 0:
            image = image.resize((width_param, height_param))

        return image, seed

def split_rectangles(rect, num, min_size):
    rectangles = [rect]
    while len(rectangles) < num:
        rect = random.choice(rectangles)
        rectangles.remove(rect)
        x1, y1, x2, y2 = rect

        if (x2 - x1) > 2 * min_size and (y2 - y1) > 2 * min_size:
            if (x2 - x1) > (y2 - y1):
                split = random.randint(x1 + min_size, x2 - min_size)
                rect1 = (x1, y1, split, y2)
                rect2 = (split, y1, x2, y2)
            else:
                split = random.randint(y1 + min_size, y2 - min_size)
                rect1 = (x1, y1, x2, split)
                rect2 = (x1, split, x2, y2)

            rectangles.extend([rect1, rect2])
        else:
            rectangles.append(rect)

    return rectangles[:num]

def resize_and_insert_image(img, rect, image):
    image = image.resize((rect[2] - rect[0], rect[3] - rect[1]))
    img.paste(image, rect[:2])

def crop_and_insert_image(img, rect, image):
    image_width, image_height = image.size
    rect_width = rect[2] - rect[0]
    rect_height = rect[3] - rect[1]

    scale_width = rect_width / image_width
    scale_height = rect_height / image_height
    scale = max(scale_width, scale_height)

    new_width = int(image_width * scale)
    new_height = int(image_height * scale)
    image = image.resize((new_width, new_height))

    left = (new_width - rect_width) // 2
    top = (new_height - rect_height) // 2
    right = left + rect_width
    bottom = top + rect_height

    cropped_image = image.crop((left, top, right, bottom))
    img.paste(cropped_image, rect[:2])

def aspect_ratio_similar(rect, image_size, tolerance=0.1):
    rect_width = rect[2] - rect[0]
    rect_height = rect[3] - rect[1]
    rect_ratio = rect_width / rect_height
    image_width, image_height = image_size
    image_ratio = image_width / image_height
    return abs(rect_ratio - image_ratio) <= tolerance

def add_metadata_file(file_path, txt_file_data):
    targetImage = Image.open(file_path)
    metadata = PngInfo()
    metadata.add_text("parameters", txt_file_data)
    targetImage.save(file_path, pnginfo=metadata)

def count_folders(directory_path_temp, new_folder):
    unique_id_temp = 0
    existing_folders = [
        int(d.split('_')[0]) for d in os.listdir(directory_path_temp) if (os.path.isdir(os.path.join(directory_path_temp, d)) and re.search(r'^\d+', d))
    ]
    if existing_folders:
        unique_id_temp = max(existing_folders)
        if new_folder:
            unique_id_temp = unique_id_temp + 1    
    else:
        unique_id_temp = 1
    return str(unique_id_temp)

def generate_manga(prompt_input, negative_prompt, width, height, margin, cfg, steps, input_seed, reverse_scenes):
    min_size_factor = 6.2
    min_size = int(width / min_size_factor)

    if prompt_input != "":
        pages = read_prompts(prompt_input, "text")
    else:
        pages = read_prompts(PROMPT_FILE, "file")
    if not pages:
        pages = "1 cat"
        
    outputs = []
    for page_num, page_prompts in enumerate(pages, 1):
        img = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(img)

        num_scenes = len(page_prompts)
        initial_rect = (margin, margin, width - margin, height - margin)
        scenes = split_rectangles(initial_rect, num_scenes, min_size)
        
        if MANGA_SCENES_ORDER:
            scenes.sort(key=lambda r: (r[1], -r[0])) # Default sort from right to left
        else:
            scenes.sort(key=lambda r: (r[1], r[0]))  # Sort from left to right

        for scene, prompt in zip(scenes, page_prompts):
            draw.rectangle(scene, outline='black', width=5)

            width_param = scene[2] - scene[0]
            height_param = scene[3] - scene[1]

            crop = False
            if width_param <= 640:
                width_param = 640
                height_param = int(640 * (width_param / height_param))
                crop = True

            [image, seed] = image_print_create(prompt, width_param, height_param, negative_prompt, cfg, steps, input_seed)
            if not crop:
                resize_and_insert_image(img, scene, image)
            else:
                crop_and_insert_image(img, scene, image)

        if MODEL_FILE_VAE == "":
            vae_string = ""
        else:
            vae_hash = calculate_sha256(os.path.join(VAE_PATH, MODEL_FILE_VAE), 10)
            vae_string = ", VAE hash: " + vae_hash + ", VAE: " + MODEL_FILE_VAE  # vae_name with extension at the end!

        txt_file_data = ""
        if MODEL_FILE_LORA[0] != "":
            i = 0
            for model_file_lora_single in MODEL_FILE_LORA:
                if float(LORA_W[i]).is_integer():
                    lora_w_number = int(LORA_W[i])
                else:
                    lora_w_number = str(LORA_W[i])
                txt_file_data = txt_file_data + f"<lora:{os.path.splitext(os.path.basename(model_file_lora_single))[0]}:{lora_w_number}>"
                i = i + 1
        else:
            txt_file_data = ""
        model_hash = calculate_sha256(os.path.join(MODEL_PATH, CHECKPOINT), 10)

        pages_str = "\n".join(page_prompts)

        txt_file_data = txt_file_data + pages_str + "\n" + "Negative prompt: " + negative_prompt + "\n" + "Steps: " + str(steps) + ", Sampler: Euler a, CFG scale: " + str(cfg) + ", Seed: " + str(seed) + ", Size: " + str(width) + "x" + str(height) + ", Model hash: " + model_hash + ", Model: " + os.path.splitext(os.path.basename(CHECKPOINT))[0] + vae_string

        if txt_file_lora != "":
            txt_file_data = txt_file_data + txt_file_lora

        print(txt_file_data)
        if not os.path.exists(MANGA_DIR):
            os.makedirs(MANGA_DIR)
        if not os.path.exists(f"./{MANGA_DIR}{current_date}"):
            os.makedirs(f"./{MANGA_DIR}{current_date}")
        if page_num == 1:
            new_folder = True
        else:
            new_folder = False    
        unique_id_folders = count_folders(f"./{MANGA_DIR}{current_date}", new_folder)
        if not os.path.exists(f"./{MANGA_DIR}{current_date}/{unique_id_folders}"):
            os.makedirs(f"./{MANGA_DIR}{current_date}/{unique_id_folders}")
        output_path = f"./{MANGA_DIR}{current_date}/{unique_id_folders}/manga_page_{page_num}.png"
        img.save(output_path)
        add_metadata_file(output_path, txt_file_data)
        outputs.append(output_path)
        print(f"saved: {output_path}")

    return outputs

if __name__ == "__main__":

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    current_datetime = date_time.now()
    current_date = current_datetime.strftime("%Y_%m_%d")

    pipeline = StableDiffusionXLPipeline.from_single_file(os.path.join(MODEL_PATH, CHECKPOINT), torch_dtype=torch_dtype_)

    if MODEL_FILE_VAE:
        vae = AutoencoderKL.from_single_file(f"{VAE_PATH}{MODEL_FILE_VAE}", torch_dtype=torch_dtype_).to(device_)
        pipeline.vae = vae

    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
    pipeline.to("cuda")

    txt_file_lora = ""  
    if MODEL_FILE_LORA[0] != "":  
        i = 0
        adapters = []
        adapter_weights = []
        for model_file_lora_single in MODEL_FILE_LORA:
            if model_file_lora_single[-12:] != ".safetensors":
                model_file_lora_single = model_file_lora_single + ".safetensors"
            i = i + 1
            pipeline.load_lora_weights(LORA_PATH, weight_name=model_file_lora_single, adapter_name=str(i))
            adapters.append(str(i))
            adapter_weights.append(float(LORA_W[i-1]))
            with open(LORA_PATH + model_file_lora_single, "rb") as file:
                if i == 1:
                    txt_file_lora = ', Lora hashes: "' + os.path.splitext(os.path.basename(model_file_lora_single))[0] + ': ' + addnet_hash_safetensors(file, 12)       
                else:
                    txt_file_lora = txt_file_lora + ", " + os.path.splitext(os.path.basename(model_file_lora_single))[0] + ': ' + addnet_hash_safetensors(file, 12)
        txt_file_lora = txt_file_lora + '"'  
        pipeline.set_adapters(adapters, adapter_weights=adapter_weights)
        # Fuses the LoRAs into the Unet
        pipeline.fuse_lora()
    else:
        txt_file_lora = ""   

    pipeline.safety_checker = None
    pipeline.requires_safety_checker = False

    iface = gr.Interface(
        fn=generate_manga,
        inputs=[
            gr.Textbox(label="Prompt", value="", lines=10),
            gr.Textbox(label="Negative Prompt", value=NEGATIVE_PROMPT),
            gr.Number(label="Width", value=WIDTH),
            gr.Number(label="Height", value=HEIGHT),
            gr.Number(label="Margin", value=MARGIN),
            gr.Number(label="CFG", value=CFG, step=1),
            gr.Number(label="Steps", value=STEPS, step=1),
            gr.Number(label="Input Seed(-1 = random)", value=INPUT_SEED, step=1, minimum=-1, maximum=9999999999),
            gr.Checkbox(label="Manga Scenes Order", value=MANGA_SCENES_ORDER)
        ],
        outputs=gr.File(label="Generated Manga Pages"),
        title="Manga Generator",
        description=f"If Prompt is empty, {PROMPT_FILE} is the default value",
        allow_flagging="never"
    )
    iface.launch(share=False, inbrowser=True)
