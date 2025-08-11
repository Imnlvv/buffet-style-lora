#!/usr/bin/env python
# coding: utf-8

# # Подгрузка модели: huggingface

# In[1]:


get_ipython().system('pip install diffusers transformers accelerate safetensors --upgrade')


# In[2]:


from diffusers import DiffusionPipeline, AutoencoderKL
import torch

# Загружаем VAE
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)

# Загружаем базовую модель SDXL
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    vae=vae,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
)

# Загружаем LoRA
repo_id = "imnlv/buffet_style_LoRA"
pipe.load_lora_weights(repo_id)

# Отправляем на GPU
pipe.to("cuda")


# # Генерации: Clown. Part 2 - Scenes with clowns

# In[8]:


# Генерация изображения
prompt = "photo collage in BUFFET style, a sad clown sitting before a cracked mirror, smudged makeup, worn-out gloves on the table. Rough, expressive lines."
image = pipe(prompt, num_inference_steps=50, guidance_scale=8.0).images[0]

# Результат
image


# In[9]:


# Генерация изображения
prompt = "photo collage in BUFFET style, a sad clown sitting before a cracked mirror, smudged makeup, worn-out gloves on the table. Rough, expressive lines."
image = pipe(prompt, num_inference_steps=50, guidance_scale=8.0).images[0]

# Результат
image


# In[10]:


# Генерация изображения
prompt = "photo collage in BUFFET style, a sad clown sitting before a cracked mirror, smudged makeup, worn-out gloves on the table. Rough, expressive lines."
image = pipe(prompt, num_inference_steps=50, guidance_scale=8.0).images[0]

# Результат
image


# In[11]:


# Генерация изображения
prompt = "photo collage in BUFFET style, a weary clown standing on an abandoned circus stage, torn curtains, dust-covered floor, holding a broken flower. Rough, expressive lines."
image = pipe(prompt, num_inference_steps=50, guidance_scale=8.0).images[0]

# Результат
image


# In[42]:


# Генерация изображения
prompt = "photo collage in BUFFET style, a lonely clown holding black balloons against a graffiti-covered brick wall, sad expression, cold urban atmosphere. Rough, expressive lines."
image = pipe(prompt, num_inference_steps=50, guidance_scale=8.0).images[0]

# Результат
image


# In[26]:


# Генерация изображения
prompt = "photo collage in BUFFET style, a lonely clown holding black balloons against a graffiti-covered brick wall, sad expression, cold urban atmosphere. Rough, expressive lines."
image = pipe(prompt, num_inference_steps=50, guidance_scale=8.0).images[0]

# Результат
image


# In[60]:


# Генерация изображения
prompt = "photo collage in BUFFET style, a lonely clown holding black balloons against a graffiti-covered brick wall, sad expression, cold urban atmosphere. Rough, expressive lines."
image = pipe(prompt, num_inference_steps=50, guidance_scale=8.0).images[0]

# Результат
image


# In[37]:


# Генерация изображения
prompt = "photo collage in BUFFET style, a sad drenched clown walking through an empty rainy street, makeup smudged, broken umbrella in hand. Rough, expressive lines."
image = pipe(prompt, num_inference_steps=50, guidance_scale=8.0).images[0]

# Результат
image


# In[39]:


# Генерация изображения
prompt = "photo collage in BUFFET style, a drenched clown walking through an empty rainy street, makeup smudged, broken umbrella in hand. Rough, expressive lines."
image = pipe(prompt, num_inference_steps=50, guidance_scale=8.0).images[0]

# Результат
image


# In[42]:


# Генерация изображения
prompt = "photo collage in BUFFET style, a drenched clown walking through an empty rainy street, makeup smudged, broken umbrella in hand. Rough, expressive lines."
image = pipe(prompt, num_inference_steps=50, guidance_scale=8.0).images[0]

# Результат
image


# In[62]:


# Генерация изображения
prompt = "photo collage in BUFFET style, a clown staring at his fragmented reflection in a broken mirror, hand on his face, rough, expressive lines, cold gray tones."
image = pipe(prompt, num_inference_steps=50, guidance_scale=8.0).images[0]

# Результат
image


# In[ ]:




