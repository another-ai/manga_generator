import random
from PIL import Image, ImageDraw
import os
import torch
import gc
from dotenv import load_dotenv
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
from diffusers import EulerAncestralDiscreteScheduler

env_path = os.path.join('env', 'manga_generator.env')
if os.path.exists(env_path):
    load_dotenv(env_path)

MAIN_DIR = os.getenv("main_dir", "./models/")
if MAIN_DIR[-1] != "/":
    MAIN_DIR = MAIN_DIR + "/"
MODEL_PATH = MAIN_DIR + os.getenv('MODEL_PATH', 'Stable-diffusion/')
CHECKPOINT = os.getenv('CHECKPOINT', 'sd_xl_base_1.0.safetensors')
VAE_PATH = MAIN_DIR + os.getenv('VAE_PATH', 'VAE/')
if VAE_PATH.startswith('/'):
    VAE_PATH = VAE_PATH[1:]
MODEL_FILE_VAE = os.getenv('MODEL_FILE_VAE', '')
LORA_PATH = MAIN_DIR + os.getenv('LORA_PATH', 'Lora/')
if LORA_PATH.startswith('/'):
    LORA_PATH = LORA_PATH[1:]
MODEL_FILE_LORA = os.getenv('MODEL_FILE_LORA', '')
NEGATIVE_PROMPT = os.getenv('NEGATIVE_PROMPT', 'blurry,blurry_image,lowres,low_resolution,low_picture_quality,low_picture_anime,extra_anatomy,extra_body,extra_navel,extra_face,extra_eyes,extra_chest,extra_nipples,extra_hips,extra_arms,extra_hands,extra_fingers,extra_legs,extra_feet,extra_toe,missing_anatomy,missing_body,missing_navel,missing_face,missing_eyes,missing_chest,missing_nipples,missing_hips,missing_arms,missing_hands,missing_fingers,missing_legs,missing_feet,missing_toe')
# A5 page dimensions in pixels (300 dpi - 1240 * 1748); default = 826, 1164; 826 = int(1240/1.5); 1164 = int(826*(1240/1748))
WIDTH = int(os.getenv('WIDTH', 826))
HEIGHT = int(os.getenv('HEIGHT', 1164))
MARGIN = int(os.getenv('MARGIN', 10))
MIN_SIZE_FACTOR = float(os.getenv('MIN_SIZE_FACTOR', 6.2))
MANGA_DIR = os.getenv('MANGA_DIR', 'manga')
PROMPT_FILE = os.getenv('PROMPT_FILE', 'manga_generator.txt')
DEFAULT_PROMPT = os.getenv('DEFAULT_PROMPT', '1 cat')

def count_file(directory_path_temp):
    unique_id_temp = 0
    existing_files = len([f for f in os.listdir(directory_path_temp) if f.endswith(".png") and os.path.isfile(os.path.join(directory_path_temp, f))])
    unique_id_temp = existing_files + 1
    return unique_id_temp

def read_prompts(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
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
    except FileNotFoundError:
        return []

def image_print_create(prompt_input, width_param, height_param):
    while True:
        prompt_ = prompt_input
        seed_ = random.randint(0, 9999999999)
        generator = torch.Generator(device=device_).manual_seed(seed_)

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

        num_inference_steps = 40
        guidance_scale = 7
        
        image = pipeline(prompt=prompt_, negative_prompt=NEGATIVE_PROMPT, generator=generator, width=width_, height=height_, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images[0]

        if resize_pixel_w > 0 or resize_pixel_h > 0:
            image = image.resize((width_param, height_param))

        print(prompt_)
        print("Negative Prompt:" + NEGATIVE_PROMPT)

        return image

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

if __name__ == "__main__":
    print("--START--")
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    device_ = "cuda"
    torch_dtype_ = torch.float16

    pipeline = StableDiffusionXLPipeline.from_single_file(os.path.join(MODEL_PATH, CHECKPOINT), torch_dtype=torch_dtype_)

    if MODEL_FILE_VAE:
        vae = AutoencoderKL.from_single_file(f"{VAE_PATH}{MODEL_FILE_VAE}", torch_dtype=torch_dtype_).to(device_)
        pipeline.vae = vae

    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
    pipeline.to("cuda")

    if MODEL_FILE_LORA:
        pipeline.load_lora_weights(LORA_PATH, weight_name=MODEL_FILE_LORA)
        pipeline.fuse_lora()

    pipeline.safety_checker = None
    pipeline.requires_safety_checker = False

    min_size = int(WIDTH / MIN_SIZE_FACTOR)

    if not os.path.exists(MANGA_DIR):
        os.makedirs(MANGA_DIR)

    pages = read_prompts(PROMPT_FILE)
    if not pages:
        pages = [[DEFAULT_PROMPT]]

    for page_num, page_prompts in enumerate(pages, 1):
        img = Image.new('RGB', (WIDTH, HEIGHT), 'white')
        draw = ImageDraw.Draw(img)

        num_scenes = len(page_prompts)
        initial_rect = (MARGIN, MARGIN, WIDTH - MARGIN, HEIGHT - MARGIN)
        scenes = split_rectangles(initial_rect, num_scenes, min_size)
        scenes.sort(key=lambda r: (r[1], -r[0]))

        for scene, prompt in zip(scenes, page_prompts):
            draw.rectangle(scene, outline='black', width=5)

            width_param = scene[2] - scene[0]
            height_param = scene[3] - scene[1]

            crop = False
            if width_param <= 640:
                width_param = 640
                height_param = int(640 * (width_param / height_param))
                crop = True

            image = image_print_create(prompt, width_param, height_param)
            if not crop:
                resize_and_insert_image(img, scene, image)
            else:
                crop_and_insert_image(img, scene, image)

        output_path = os.path.join(MANGA_DIR, f"manga_page_{page_num}.png")
        print(f"saved: {output_path}")
        img.save(output_path)
