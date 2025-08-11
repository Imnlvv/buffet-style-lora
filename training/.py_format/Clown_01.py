#!/usr/bin/env python
# coding: utf-8

# # Clown. Part 1

# ## Установка 🪓

# #### 01
# Проверяем GPU и скачиваем зависимости:

# In[1]:


get_ipython().system('nvidia-smi')


# In[2]:


get_ipython().system('pip install bitsandbytes transformers accelerate peft -q')


# In[3]:


get_ipython().system('pip install git+https://github.com/huggingface/diffusers.git -q')


# #### 02
# Загружаем обучающий скрипт DreamBooth для диффузоров SDXL:

# In[4]:


get_ipython().system('wget https://raw.githubusercontent.com/huggingface/diffusers/main/examples/dreambooth/train_dreambooth_lora_sdxl.py')


# ## Датасет 🐶

# #### 01
# Создаем датасет и загружаем фото:

# In[5]:


get_ipython().system('mkdir -p buff')
get_ipython().system('cp /kaggle/input/bernard-buffet-arts/* buff/')


# In[6]:


from PIL import Image

def image_grid(imgs, rows, cols, resize=256):

    if resize is not None:
        imgs = [img.resize((resize, resize)) for img in imgs]
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


# In[8]:


import glob

img_paths = "./buff/*.png"
imgs = [Image.open(path) for path in glob.glob(img_paths)]

num_imgs_to_preview = 5
image_grid(imgs[:num_imgs_to_preview], 1, num_imgs_to_preview)


# #### 02
# Создаем собственные подписи к изображениям с помощью BLIP:

# In[9]:


import requests
from transformers import AutoProcessor, BlipForConditionalGeneration
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# Загрузка процессора и модели субтитров
blip_processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base",torch_dtype=torch.float16).to(device)

# Утилита для субтитров
def caption_images(input_image):
    inputs = blip_processor(images=input_image, return_tensors="pt").to(device, torch.float16)
    pixel_values = inputs.pixel_values

    generated_ids = blip_model.generate(pixel_values=pixel_values, max_length=50)
    generated_caption = blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_caption


# In[10]:


import glob
from PIL import Image

# Создаем список пар (Pil.Image, path)
local_dir = "./buff/"
imgs_and_paths = [(path,Image.open(path)) for path in glob.glob(f"{local_dir}*.png")]


# In[12]:


import json

caption_prefix = "photo collage in BUFFET style, "
with open(f'{local_dir}metadata.jsonl', 'w') as outfile:
  for img in imgs_and_paths:
      caption = caption_prefix + caption_images(img[1]).split("\n")[0]
      entry = {"file_name":img[0].split("/")[-1], "prompt": caption}
      json.dump(entry, outfile)
      outfile.write('\n')


# In[13]:


get_ipython().system('cat buff/metadata.jsonl')


# In[14]:


import gc

del blip_processor, blip_model
gc.collect()
torch.cuda.empty_cache()


# ## Подготовка к обучению 💻

# #### 01
# Инициализируем `accelerate`:

# In[16]:


import locale
locale.getpreferredencoding = lambda: "UTF-8"

get_ipython().system('accelerate config default')


# #### 02
# Входим в учетную запись Hugging Face:

# In[17]:


from huggingface_hub import notebook_login
notebook_login()


# ## Обучение модели! 🔬

# #### 01
# Задаем все значения и обучаем модель `buffet_style_LoRA`:

# In[18]:


get_ipython().system('pip install datasets -q')


# In[19]:


#!/usr/bin/env bash
get_ipython().system('accelerate launch train_dreambooth_lora_sdxl.py    --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0"    --pretrained_vae_model_name_or_path="madebyollin/sdxl-vae-fp16-fix"    --dataset_name="buff"    --output_dir="buffet_style_LoRA"    --caption_column="prompt"   --mixed_precision="fp16"    --instance_prompt="photo collage in BUFFET style"    --resolution=1024    --train_batch_size=1    --gradient_accumulation_steps=3    --gradient_checkpointing    --learning_rate=1e-4    --snr_gamma=5.0    --lr_scheduler="constant"    --lr_warmup_steps=0    --mixed_precision="fp16"    --use_8bit_adam    --max_train_steps=1000    --checkpointing_steps=500    --seed="0"')


# In[20]:


get_ipython().system('ls buffet_style_LoRA')


# #### 02
# Сохраняем модель в hub и проверяем ее:

# In[21]:


from huggingface_hub import whoami
from pathlib import Path

output_dir = "buffet_style_LoRA"
username = whoami(token=Path("/root/.cache/huggingface/"))["name"]
repo_id = f"{username}/{output_dir}"


# In[22]:


from train_dreambooth_lora_sdxl import save_model_card
from huggingface_hub import upload_folder, create_repo

repo_id = create_repo(repo_id, exist_ok=True).repo_id

save_model_card(
    repo_id = repo_id,
    images=[],
    base_model="stabilityai/stable-diffusion-xl-base-1.0",
    train_text_encoder=False,
    instance_prompt="photo collage in BUFFET style",
    validation_prompt=None,
    repo_folder=output_dir,
    use_dora = False,
    vae_path="madebyollin/sdxl-vae-fp16-fix",
)

upload_folder(
    repo_id=repo_id,
    folder_path=output_dir,
    commit_message="End of training",
    ignore_patterns=["step_*", "epoch_*"],
)


# In[23]:


from IPython.display import display, Markdown

link_to_model = f"https://huggingface.co/{repo_id}"
display(Markdown("### Your model has finished training.\nAccess it here: {}".format(link_to_model)))


# ## Генерация 🤡

# #### 01
# Загружаем библиотеку с huggingface - `buffet_style_LoRA`:

# In[1]:


import torch
from diffusers import DiffusionPipeline, AutoencoderKL

vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    vae=vae,
    torch_dtype=torch.bfloat16,
    variant="fp16",
    use_safetensors=True
)
pipe.load_lora_weights(repo_id)
_ = pipe.to("cuda")


# In[2]:


from diffusers import DiffusionPipeline, AutoencoderKL
import torch

# Загружаем VAE
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)

# Загружаем базовую модель Stable Diffusion XL
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    vae=vae,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
)

# Загружаем LoRA-веса напрямую с Hugging Face
pipe.load_lora_weights("imnlv/buffet_style_LoRA", revision="main")

# Применяем LoRA
pipe.fuse_lora(lora_scale=1.0)

# Перемещаем модель на GPU
pipe.to("cuda")


# #### 02
# Генерируем изображения: Clown. Part 01 - `Clown portraits`:

# In[8]:


# Генерация изображения
prompt = "photo collage in BUFFET style, portrait of a clown who grins, green and yellow"
image = pipe(prompt, num_inference_steps=50, guidance_scale=8.0).images[0]

# Результат
image


# In[21]:


# Генерация изображения
prompt = "photo collage in BUFFET style, portrait of a clown who sad, blue and black"
image = pipe(prompt, num_inference_steps=50, guidance_scale=8.0).images[0]

# Результат
image


# In[24]:


# Генерация изображения
prompt = "photo collage in BUFFET style, portrait of a clown who sad and cry, blue and black"
image = pipe(prompt, num_inference_steps=50, guidance_scale=8.0).images[0]

# Результат
image


# In[13]:


# Генерация изображения
prompt = "photo collage in BUFFET style, portrait of a clown who smile, white and yellow"
image = pipe(prompt, num_inference_steps=50, guidance_scale=8.0).images[0]

# Результат
image


# In[14]:


# Генерация изображения
prompt = "photo collage in BUFFET style, portrait of a clown who angry and mad, red and black"
image = pipe(prompt, num_inference_steps=50, guidance_scale=8.0).images[0]

# Результат
image


# In[32]:


# Генерация изображения
prompt = "photo collage in BUFFET style, portrait of a clown who is surprised and shocked, brown and yellow"
image = pipe(prompt, num_inference_steps=50, guidance_scale=8.0).images[0]

# Результат
image


# In[54]:


# Генерация изображения
prompt = "photo collage in BUFFET style, portrait of a clown who is very afraid, panicking and crying, black and white"
image = pipe(prompt, num_inference_steps=50, guidance_scale=8.0).images[0]

# Результат
image


# In[4]:


# Генерация изображения
prompt = "photo collage in BUFFET style, portrait of a clown who feels admiration and love, white and pink"
image = pipe(prompt, num_inference_steps=50, guidance_scale=8.0).images[0]

# Результат
image


# In[9]:


import nbformat

with open("Clown_01.ipynb") as f:
    nb = nbformat.read(f, as_version=4)

nb.metadata.pop("widgets", None)

with open("Clown_01.ipynb", "w") as f:
    nbformat.write(nb, f)


# In[ ]:




