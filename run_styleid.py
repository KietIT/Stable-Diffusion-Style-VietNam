#!/usr/bin/env python
"""
StyleID - Style Injection in Diffusion
Optimized for VRAM Efficiency (CPU Offloading) & ComfyUI Integration
Final Fix: OOM, Device Mismatch, Randomness Logic
"""
# 1. CẤU HÌNH CHỐNG PHÂN MẢNH VRAM NGAY LẬP TỨC
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

import random
import argparse
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from einops import rearrange
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext
import copy
import time
import pickle
import torchvision.transforms as transforms

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

# --- BIẾN TOÀN CỤC ---
feat_maps = []
idx_time_dict = {}
time_idx_dict = {}

# --- CÁC HÀM HỖ TRỢ ---

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:", m)
    if len(u) > 0 and verbose:
        print("unexpected keys:", u)

    # Load thẳng vào CUDA
    model.cuda()
    model.eval()
    return model

def load_img(path):
    image = Image.open(path).convert("RGB")
    x, y = image.size
    print(f"Loaded input image of size ({x}, {y}) from {path}")
    h = w = 512
    image = transforms.CenterCrop(min(x,y))(image)
    image = image.resize((w, h), resample=Image.Resampling.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.

def adain(cnt_feat, sty_feat):
    # Đảm bảo tính toán
    cnt_mean = cnt_feat.mean(dim=[0, 2, 3], keepdim=True)
    cnt_std = cnt_feat.std(dim=[0, 2, 3], keepdim=True)
    sty_mean = sty_feat.mean(dim=[0, 2, 3], keepdim=True)
    sty_std = sty_feat.std(dim=[0, 2, 3], keepdim=True)
    
    output = ((cnt_feat - cnt_mean) / (cnt_std + 1e-6)) * sty_std + sty_mean
    return output

def feat_merge(opt, cnt_feats, sty_feats, start_step=0):
    """
    Trộn đặc trưng và QUAN TRỌNG: Đưa Tensor từ CPU lên GPU để Model xử lý
    """
    # Tự động xác định thiết bị (thường là cuda)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    merged_maps = [{'config': {
                'gamma': opt.gamma,
                'T': opt.T,
                'timestep': _,
                }} for _ in range(50)]

    for i in range(len(merged_maps)):
        if i < (50 - start_step):
            continue
        
        cnt_feat = cnt_feats[i]
        sty_feat = sty_feats[i]
        
        # Lấy hợp tập tất cả các key (q, k, v...)
        all_keys = set(cnt_feat.keys()) | set(sty_feat.keys())

        for key in all_keys:
            if key == 'config': continue
            
            # Logic trộn: 'q' lấy từ content, 'k' & 'v' lấy từ style
            if key.endswith('q'):
                val = cnt_feat.get(key)
            elif key.endswith('k') or key.endswith('v'):
                val = sty_feat.get(key)
            else:
                val = sty_feat.get(key)

            if val is not None:
                # CƯỠNG ÉP LÊN GPU: Nếu đang ở CPU (do bước save), phải .to(device)
                if isinstance(val, torch.Tensor):
                    merged_maps[i][key] = val.to(device)
                else:
                    merged_maps[i][key] = val
                    
    return merged_maps

# --- CÁC HÀM CALLBACK LƯU FEATURE (ĐẨY SANG CPU) ---

def save_feature_map(feature_map, filename, time):
    global feat_maps
    if time in idx_time_dict:
        cur_idx = idx_time_dict[time]
        # FIX OOM: Đẩy sang CPU ngay lập tức
        feat_maps[cur_idx][f"{filename}"] = feature_map.detach().cpu()

def save_feature_maps(blocks, i, feature_type, indices):
    for block_idx, block in enumerate(blocks):
        if len(block) > 1 and "SpatialTransformer" in str(type(block[1])):
            if block_idx in indices:
                q = block[1].transformer_blocks[0].attn1.q
                k = block[1].transformer_blocks[0].attn1.k
                v = block[1].transformer_blocks[0].attn1.v
                save_feature_map(q, f"{feature_type}_{block_idx}_self_attn_q", i)
                save_feature_map(k, f"{feature_type}_{block_idx}_self_attn_k", i)
                save_feature_map(v, f"{feature_type}_{block_idx}_self_attn_v", i)

# --- MAIN ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cnt', default='./data/cnt', help='Content path (file or folder)')
    parser.add_argument('--sty', default='./data/sty', help='Style path (file or folder)')
    parser.add_argument('--ddim_inv_steps', type=int, default=50)
    parser.add_argument('--save_feat_steps', type=int, default=50)
    parser.add_argument('--start_step', type=int, default=49)
    parser.add_argument('--ddim_eta', type=float, default=0.0)
    parser.add_argument('--H', type=int, default=512)
    parser.add_argument('--W', type=int, default=512)
    parser.add_argument('--output_H', type=int, default=1024)
    parser.add_argument('--output_W', type=int, default=1024)
    parser.add_argument('--C', type=int, default=4)
    parser.add_argument('--f', type=int, default=8)
    parser.add_argument('--T', type=float, default=1.5)
    parser.add_argument('--gamma', type=float, default=0.75)
    parser.add_argument("--attn_layer", type=str, default='6,7,8,9,10,11')
    parser.add_argument('--model_config', type=str, default='models/ldm/stable-diffusion-v1/v1-inference.yaml')
    parser.add_argument('--precomputed', type=str, default='')
    parser.add_argument('--ckpt', type=str, default='models/ldm/stable-diffusion-v1/model.ckpt')
    parser.add_argument('--precision', type=str, default='autocast')
    parser.add_argument('--output_path', type=str, default='output')
    parser.add_argument("--without_init_adain", action='store_true')
    parser.add_argument("--without_attn_injection", action='store_true')
    parser.add_argument("--low_vram_mode", action='store_true')
    
    opt = parser.parse_args()

    # Tạo thư mục output
    os.makedirs(opt.output_path, exist_ok=True)
    if len(opt.precomputed) > 0:
        os.makedirs(opt.precomputed, exist_ok=True)

    # [QUAN TRỌNG] Seed cố định cho Diffusion (giữ nét vẽ ổn định)
    # Lệnh này sẽ được giữ lại để đảm bảo tính nhất quán của model
    seed_everything(22)

    print("=" * 60)
    print("StyleID - Optimized for VRAM & ComfyUI")
    print("=" * 60)

    # Load Model
    model_config = OmegaConf.load(f"{opt.model_config}")
    model = load_model_from_config(model_config, f"{opt.ckpt}")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Optimization
    if opt.low_vram_mode:
        print("[OPTIMIZATION] Enabling gradient checkpointing...")
        if hasattr(model.model.diffusion_model, 'enable_gradient_checkpointing'):
            model.model.diffusion_model.enable_gradient_checkpointing()
        torch.cuda.empty_cache()

    # Setup Sampler & Parameters
    unet_model = model.model.diffusion_model
    self_attn_output_block_indices = list(map(int, opt.attn_layer.split(',')))
    
    sampler = DDIMSampler(model)
    sampler.make_schedule(ddim_num_steps=opt.save_feat_steps, ddim_eta=opt.ddim_eta, verbose=False)
    
    # Tạo mapping thời gian
    time_range = np.flip(sampler.ddim_timesteps)
    global idx_time_dict, time_idx_dict
    idx_time_dict = {t: i for i, t in enumerate(time_range)}
    time_idx_dict = {i: t for i, t in enumerate(time_range)}

    # --- CALLBACKS DEFINITION ---
    def ddim_sampler_callback(pred_x0, xt, i):
        save_feature_maps(unet_model.output_blocks, i, "output_block", self_attn_output_block_indices)
        # Lưu latent xt sang CPU
        save_feature_map(xt, 'z_enc', i)

    # --- PREPARE IMAGES (LOGIC SỬA ĐỔI ĐỂ HỖ TRỢ RANDOM TỪ COMYUI) ---
    
    # 1. Xử lý Style Path (Hỗ trợ File hoặc Folder)
    if os.path.isfile(opt.sty):
        # Nếu ComfyUI truyền vào 1 file cụ thể (đã random ở node), ta dùng file đó
        sty_img_list = [os.path.basename(opt.sty)]
        sty_dir = os.path.dirname(opt.sty)
    else:
        # Fallback: Nếu truyền folder, lấy danh sách (bị ảnh hưởng bởi seed 22)
        sty_img_list = sorted([f for f in os.listdir(opt.sty) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        sty_dir = opt.sty

    # 2. Xử lý Content Path
    if os.path.isfile(opt.cnt):
        cnt_img_list = [os.path.basename(opt.cnt)]
        cnt_dir = os.path.dirname(opt.cnt)
    else:
        cnt_img_list = sorted([f for f in os.listdir(opt.cnt) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        cnt_dir = opt.cnt

    if not sty_img_list:
        print(f"❌ Error: Không tìm thấy ảnh style nào tại {opt.sty}")
        return
    if not cnt_img_list:
        print(f"❌ Error: Không tìm thấy ảnh content nào tại {opt.cnt}")
        return

    # Khởi tạo biến toàn cục feat_maps
    global feat_maps
    
    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    uc = model.get_learned_conditioning([""]).to(device) # UC phải ở GPU
    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]

    begin_time = time.time()

    # --- MAIN LOOP ---
    for sty_name in sty_img_list:
        sty_path_full = os.path.join(sty_dir, sty_name)
        print(f"🎨 Processing Style: {sty_name}")

        # 1. ENCODE STYLE
        # Load ảnh -> GPU
        init_sty = load_img(sty_path_full).to(device)
        
        # Reset feat_maps
        feat_maps = [{'config': {'gamma': opt.gamma, 'T': opt.T}} for _ in range(50)]
        
        with torch.no_grad(), precision_scope("cuda"):
            init_sty = model.get_first_stage_encoding(model.encode_first_stage(init_sty))
            
            # Chạy Inversion
            _, _ = sampler.encode_ddim(init_sty.clone(), num_steps=opt.ddim_inv_steps, unconditional_conditioning=uc,
                                       end_step=time_idx_dict[opt.ddim_inv_steps-1-opt.start_step],
                                       callback_ddim_timesteps=opt.save_feat_steps,
                                       img_callback=ddim_sampler_callback)
            
            # Copy features (Lúc này features đã nằm trên CPU nhờ hàm save_feature_map)
            # Dùng deepcopy là an toàn vì dữ liệu ở CPU, không tốn VRAM
            sty_feat = copy.deepcopy(feat_maps)
            # Lấy z_enc và đảm bảo nó lên GPU cho bước tiếp theo (nếu cần dùng ngay)
            # Nhưng ở đây z_enc lưu trong feat_maps[0] đang là CPU.
            sty_z_enc = feat_maps[0]['z_enc'].to(device) 

        for cnt_name in cnt_img_list:
            cnt_path_full = os.path.join(cnt_dir, cnt_name)
            print(f"   🖼️  Content: {cnt_name}")

            # 2. ENCODE CONTENT
            init_cnt = load_img(cnt_path_full).to(device)
            feat_maps = [{'config': {'gamma': opt.gamma, 'T': opt.T}} for _ in range(50)]

            with torch.no_grad(), precision_scope("cuda"):
                init_cnt = model.get_first_stage_encoding(model.encode_first_stage(init_cnt))
                _, _ = sampler.encode_ddim(init_cnt.clone(), num_steps=opt.ddim_inv_steps, unconditional_conditioning=uc,
                                           end_step=time_idx_dict[opt.ddim_inv_steps-1-opt.start_step],
                                           callback_ddim_timesteps=opt.save_feat_steps,
                                           img_callback=ddim_sampler_callback)
                
                cnt_feat = copy.deepcopy(feat_maps)
                cnt_z_enc = feat_maps[0]['z_enc'].to(device)

            if opt.low_vram_mode:
                torch.cuda.empty_cache()

            # 3. GENERATE (INFERENCE)
            print("      🚀 Generating...")
            with torch.no_grad(), precision_scope("cuda"):
                with model.ema_scope():
                    output_name = f"{os.path.basename(cnt_name).split('.')[0]}_stylized_{os.path.basename(sty_name).split('.')[0]}.png"

                    # AdaIN (Đầu vào là GPU, Đầu ra là GPU)
                    if opt.without_init_adain:
                        adain_z_enc = cnt_z_enc
                    else:
                        adain_z_enc = adain(cnt_z_enc, sty_z_enc)

                    # Merge Features (Hàm này sẽ tự động đưa CPU features lên GPU)
                    merged_feat_maps = feat_merge(opt, cnt_feat, sty_feat, start_step=opt.start_step)
                    
                    if opt.without_attn_injection:
                        merged_feat_maps = None

                    # Sampling
                    samples_ddim, _ = sampler.sample(
                        S=opt.save_feat_steps,
                        batch_size=1,
                        shape=shape,
                        verbose=False,
                        unconditional_conditioning=uc,
                        eta=opt.ddim_eta,
                        x_T=adain_z_enc,
                        injected_features=merged_feat_maps, # Đã lên GPU
                        start_step=opt.start_step
                    )

                    # Decode & Save
                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
                    x_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)
                    x_sample = 255. * rearrange(x_image_torch[0].cpu().numpy(), 'c h w -> h w c')
                    img = Image.fromarray(x_sample.astype(np.uint8))

                    # UPSCALING (LANCZOS)
                    target_size = (opt.output_W, opt.output_H)
                    if img.size != target_size:
                        print(f"      📈 Upscaling to {target_size}")
                        img = img.resize(target_size, resample=Image.Resampling.LANCZOS)

                    output_filepath = os.path.join(opt.output_path, output_name)
                    img.save(output_filepath)
                    print(f"      ✅ Saved: {output_filepath}")

            # Dọn dẹp bộ nhớ sau mỗi vòng lặp
            del merged_feat_maps, cnt_feat, cnt_z_enc
            if opt.low_vram_mode:
                torch.cuda.empty_cache()

    print(f"🏁 Task completed in {time.time() - begin_time:.2f}s")

if __name__ == "__main__":
    main()
