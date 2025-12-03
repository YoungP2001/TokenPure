
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["http_proxy"] = "http://127.0.0.1:17890"
# os.environ["https_proxy"] = "http://127.0.0.1:17890"
import re
import glob
import numpy as np
from PIL import Image
import json
from tqdm import tqdm  # for displaying progress bar

import torch
import torch.nn as nn

from src.pipeline_flux import FluxPipeline
from src.transformer_flux import FluxTransformer2DModel
from src.layers import MultiDoubleStreamBlockLoraProcessor, MultiSingleStreamBlockLoraProcessor
from transformers import AutoProcessor, SiglipVisionModel
from src.lora_helper import load_checkpoint, \
    update_model_with_lora_and_ip_adapter3


# ======================== Image Resize Function ===========================
def resize_img(input_image, pad_to_regular=False, target_long_side=512, mode=Image.BILINEAR):
    w, h = input_image.size
    aspect_ratios = [(3, 4), (4, 3), (1, 1), (16, 9), (9, 16)]

    if pad_to_regular:
        img_ratio = w / h

        # Find the aspect ratio closest to the original image
        best_ratio = min(
            aspect_ratios,
            key=lambda r: abs((r[0] / r[1]) - img_ratio)
        )

        target_w_ratio, target_h_ratio = best_ratio
        if w / h >= target_w_ratio / target_h_ratio:
            target_w = w
            target_h = int(w * target_h_ratio / target_w_ratio)
        else:
            target_h = h
            target_w = int(h * target_w_ratio / target_h_ratio)

        # Create white background and paste the image centered
        padded_img = Image.new("RGB", (target_w, target_h), (255, 255, 255))
        offset_x = (target_w - w) // 2
        offset_y = (target_h - h) // 2
        padded_img.paste(input_image, (offset_x, offset_y))
        input_image = padded_img
        w, h = input_image.size

    # Resize while keeping aspect ratio
    scale_ratio = target_long_side / max(w, h)
    new_w = round(w * scale_ratio)
    new_h = round(h * scale_ratio)
    input_image = input_image.resize((new_w, new_h), mode)

    return input_image


# ======================== MLP Projection Module ===========================
class MLPProjModel(torch.nn.Module):
    def __init__(self, cross_attention_dim=768, id_embeddings_dim=512, num_tokens=4):
        super().__init__()

        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens

        self.proj = torch.nn.Sequential(
            torch.nn.Linear(id_embeddings_dim, id_embeddings_dim * 2),
            torch.nn.GELU(),
            torch.nn.Linear(id_embeddings_dim * 2, cross_attention_dim),
        )
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, id_embeds):
        x = self.proj(id_embeds)
        # x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)
        x = self.norm(x)
        return x


# ======================== IPAdapter Wrapper ===========================
class IPAdapter:
    def __init__(self, sd_pipe, image_encoder_path, lora_ckpt, ip_ckpt, device, num_tokens=4, cond_size=128):
        self.device = device
        self.image_encoder_path = image_encoder_path
        self.ip_ckpt = ip_ckpt
        self.lora_ckpt = lora_ckpt
        self.num_tokens = num_tokens
        self.cond_size = cond_size
        self.lora_weights = [1.0] * 1

        self.pipe = sd_pipe.to(self.device)
        # self.set_ip_adapter()

        # Load image encoder
        self.image_encoder = SiglipVisionModel.from_pretrained(image_encoder_path).to(self.device, dtype=torch.bfloat16)
        self.clip_image_processor = AutoProcessor.from_pretrained(self.image_encoder_path)

        # Initialize image projection model
        self.image_proj_model = self.init_proj()
        self.image_proj_model.eval()

        # self.load_ip_adapter()
        self.set_lora_and_ip_adapter(scale=1)

    def init_proj(self):
        image_proj_model = MLPProjModel(
            cross_attention_dim=self.pipe.transformer.config.joint_attention_dim,  # 4096
            id_embeddings_dim=1152,
            num_tokens=self.num_tokens,
        ).to(self.device, dtype=torch.bfloat16)

        return image_proj_model

    def set_ip_adapter(self, scale=1):
        n_loras = 1
        ranks = [128] * n_loras
        lora_weights = [1.0] * n_loras
        network_alphas = ranks

        transformer = self.pipe.transformer
        lora_attn_procs = {}
        double_blocks_idx = list(range(19))
        single_blocks_idx = list(range(38))
        for name, attn_processor in transformer.attn_processors.items():
            match = re.search(r'\.(\d+)\.', name)
            if match:
                layer_index = int(match.group(1))
            if name.startswith("transformer_blocks") and layer_index in double_blocks_idx:
                print("setting LoRA Processor for", name)
                lora_attn_procs[name] = MultiDoubleStreamBlockLoraProcessor(
                    dim=3072, ranks=ranks, network_alphas=network_alphas,
                    lora_weights=[1 for _ in range(n_loras)], device=self.device, dtype=torch.bfloat16,
                    cond_width=self.cond_size, cond_height=self.cond_size, n_loras=n_loras,
                    cross_attention_dim=transformer.config.joint_attention_dim,
                    heads=24, scale=scale
                )
            elif name.startswith("single_transformer_blocks") and layer_index in single_blocks_idx:
                print("setting LoRA Processor for", name)
                lora_attn_procs[name] = MultiSingleStreamBlockLoraProcessor(
                    dim=3072, ranks=ranks, network_alphas=network_alphas,
                    lora_weights=[1 for _ in range(n_loras)], device=self.device, dtype=torch.bfloat16,
                    cond_width=self.cond_size, cond_height=self.cond_size, n_loras=n_loras,
                    cross_attention_dim=transformer.config.joint_attention_dim,
                    heads=24, scale=scale
                )
            else:
                lora_attn_procs[name] = attn_processor

        transformer.set_attn_processor(lora_attn_procs)

    def load_ip_adapter(self):
        state_dict = torch.load(self.ip_ckpt, map_location="cpu")

        # ---- 1. Load weights for image_proj_model ----
        image_proj_state_dict = state_dict["image_proj"]
        if list(image_proj_state_dict.keys())[0].startswith("module."):
            # Remove DataParallel/DDP prefix
            image_proj_state_dict = {
                k.replace("module.", ""): v for k, v in image_proj_state_dict.items()
            }
        self.image_proj_model.load_state_dict(image_proj_state_dict, strict=True)

        # ---- 2. Load weights for ip_adapter (attn_processors) ----
        ip_adapter_state_dict = state_dict["ip_adapter"]
        if list(ip_adapter_state_dict.keys())[0].startswith("module."):
            ip_adapter_state_dict = {
                k.replace("module.", ""): v for k, v in ip_adapter_state_dict.items()
            }

        ip_layers = torch.nn.ModuleList(self.pipe.transformer.attn_processors.values())
        ip_layers.load_state_dict(ip_adapter_state_dict, strict=False)

    def set_lora_and_ip_adapter(self, scale=1):

        transformer = self.pipe.transformer
        lora_checkpoint = load_checkpoint(self.lora_ckpt)
        ip_adapter_checkpoint = load_checkpoint(self.ip_ckpt)

        update_model_with_lora_and_ip_adapter3(
            checkpoint=lora_checkpoint,
            lora_weights=self.lora_weights,
            ip_adapter_checkpoint=ip_adapter_checkpoint,
            transformer=transformer,
            mlp_proj_model=self.image_proj_model,
            cond_size=self.cond_size,
            device=self.device,
            scale=scale
        )

        print(f"Successfully loaded LoRA ({self.lora_ckpt}) and IP-Adapter ({self.ip_ckpt})")

    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
        if pil_image is not None:
            if isinstance(pil_image, Image.Image):
                pil_image = [pil_image]
            clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
            clip_image_embeds = self.image_encoder(
                clip_image.to(self.device, dtype=self.image_encoder.dtype)).last_hidden_state
            clip_image_embeds = clip_image_embeds.to(dtype=torch.bfloat16)
        else:
            clip_image_embeds = clip_image_embeds.to(self.device, dtype=torch.bfloat16)
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        return image_prompt_embeds

    def generate(
            self,
            strength=0.6,
            height=256,
            width=256,
            cond_size=128,
            img_path=None,  # supports list or tuple of two PIL images
            prompt=None,
            scale=1.0,
            num_samples=1,
            seed=None,
            guidance_scale=1.5,
            num_inference_steps=24,
            max_seq_len=128,
            **kwargs,
    ):

        pil_image = Image.open(img_path).convert("RGB")


        tqdm.write(f"Original image size: {pil_image.size} (w x h)")
        pil_image = resize_img(
            input_image=pil_image,
            pad_to_regular=True,
            target_long_side=cond_size
        )
        # ------------------------------------------------------------------------------

        image_prompt_embeds = self.get_image_embeds(
            pil_image=pil_image, clip_image_embeds=None
        )

        from controlnet_aux import CannyDetector
        canny_processor = CannyDetector()  #
        image_np = np.array(pil_image)
        canny_image = canny_processor(image_np, low_threshold=100, high_threshold=150)
        canny_image_pil = Image.fromarray(canny_image)

        generator = torch.Generator(self.device).manual_seed(seed) if seed is not None else None

        # image_prompt_embeds=None

        images = self.pipe(
            prompt=prompt,
            image=pil_image,
            subject_images=[],
            spatial_images=[canny_image_pil],
            image_emb=image_prompt_embeds,
            height=height,
            width=width,
            strength=strength,
            cond_size=cond_size,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            max_sequence_length=max_seq_len,
            generator=generator,
            **kwargs,
        )

        return images


# ======================== Parameter Setup ===========================
BASE_MODEL_PATH = ".../models_black_forest_labs_FLUX.1_dev"
IMAGE_ENCODER_PATH = ".../models_google_siglip_so400m_patch14_384"

IPADAPTER_PATH = ".../ip_adapter.safetensors"
LORA_WEIGHTS_PATH  = ".../lora.safetensors"

DEVICE = "cuda"


input_dir = '/tokenpure/test_data/1013_test_512'
prompt_dir = '/tokenpure/test_data/1013_test_512'
output_dir = "/tokenpure/test_data/recon_out_1013_512"

os.makedirs(output_dir, exist_ok=True)
# ======================== Model Setup ===========================
transformer = FluxTransformer2DModel.from_pretrained(
    BASE_MODEL_PATH, subfolder="transformer", torch_dtype=torch.bfloat16
)
pipe = FluxPipeline.from_pretrained(
    BASE_MODEL_PATH, transformer=transformer, torch_dtype=torch.bfloat16
)

height = 512
width = height
cond_size = 512
guidance_scale = 3.5
num_inference_steps = 25
max_seq_len = 256
strength = 1



ip_model = IPAdapter(pipe, IMAGE_ENCODER_PATH, LORA_WEIGHTS_PATH, IPADAPTER_PATH, device=DEVICE, num_tokens=128,
                     cond_size=cond_size)
# ip_model.eval()
print("IP-Adapter initialized ✔️")

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff")

files = sorted(f for f in os.listdir(input_dir)
               if f.lower().endswith(IMAGE_EXTENSIONS))
for fn in tqdm(files, desc="Inferring"):
    base = os.path.splitext(fn)[0]
    img_path = os.path.join(input_dir, fn)
    prompt_txt = os.path.join(prompt_dir, base + ".txt")
    out_path = os.path.join(output_dir, fn)

    if not os.path.isfile(prompt_txt):
        raise FileNotFoundError(f"Prompt file not found: {prompt_txt}")
    prompt = open(prompt_txt, "r", encoding="utf-8").read().strip()

    generated_images = ip_model.generate(
        strength=strength,
        img_path=img_path,
        prompt=prompt,
        height=height,
        width=width,
        cond_size=cond_size,
        scale=1.0,
        seed=1000,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        max_seq_len=max_seq_len,
    )

    out_img = generated_images.images[0]
    out_img.save(out_path)
    tqdm.write(f"Saved → {out_path}")
