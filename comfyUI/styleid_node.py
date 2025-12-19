import os
import subprocess
import torch
import numpy as np
from PIL import Image
import time
import shutil
import random
import uuid

# Danh sách Style khớp với folder data/sty/
STYLE_LIST = ["son_dau", "son_mai", "dong_ho", "hang_trong", "khac_go", "lua", "lang_sinh_hue", "dan_toc_thieu_so"]

class StyleID_Wrapper_Node:
    def __init__(self): pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "content_image": ("IMAGE",), 
                "style_name": (STYLE_LIST,),
                "python_path": ("STRING", {"default": "/opt/conda/envs/styleid_env/bin/python"}),
                "project_root": ("STRING", {"default": "/workspace/Stable-Diffusion-Style-VietNam"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run_styleid"
    CATEGORY = "HeritageArt"

    def run_styleid(self, content_image, style_name, python_path, project_root):
        torch.cuda.empty_cache()
        project_root = os.path.abspath(project_root)
        
        # 1. CLEANUP: Xóa sạch folder cũ để tránh "ảnh ma"
        temp_input_dir = os.path.join(project_root, "temp_user_input")
        temp_style_dir = os.path.join(project_root, "temp_style_input")
        output_dir = os.path.join(project_root, "temp_output")

        for folder in [temp_input_dir, temp_style_dir, output_dir]:
            if os.path.exists(folder):
                shutil.rmtree(folder, ignore_errors=True)
            os.makedirs(folder, exist_ok=True)

        # 2. CHỌN STYLE NGẪU NHIÊN
        data_dir = os.path.join(project_root, "data", "sty", style_name)
        try:
            valid_ext = ('.png', '.jpg', '.jpeg')
            all_imgs = [f for f in os.listdir(data_dir) if f.lower().endswith(valid_ext)]
            if not all_imgs: raise Exception("Folder style rỗng!")
            chosen = random.choice(all_imgs)
            shutil.copy2(os.path.join(data_dir, chosen), os.path.join(temp_style_dir, chosen))
        except Exception as e: raise Exception(f"Lỗi Style: {e}")

        # 3. LƯU ẢNH INPUT (Tên UUID để tránh Cache browser)
        unique_name = f"input_{uuid.uuid4().hex[:8]}.png"
        img_tensor = content_image[0].cpu().numpy() * 255.0
        input_pil = Image.fromarray(np.clip(img_tensor, 0, 255).astype(np.uint8))
        input_pil.save(os.path.join(temp_input_dir, unique_name))

        # 4. CẤU HÌNH LỆNH CHẠY
        cmd = [
            python_path, os.path.join(project_root, "run_styleid.py"),
            "--cnt", temp_input_dir,
            "--sty", temp_style_dir, 
            "--output_path", output_dir,
            "--model_config", os.path.join(project_root, "models/ldm/stable-diffusion-v1/v1-inference.yaml"),
            "--ckpt", os.path.join(project_root, "models/ldm/stable-diffusion-v1/model.ckpt"),
            "--low_vram_mode"
        ]
        
        # 5. FIX LỖI CUDA ALLOCATOR (QUAN TRỌNG)
        my_env = os.environ.copy()
        if "PYTORCH_CUDA_ALLOC_CONF" in my_env:
            del my_env["PYTORCH_CUDA_ALLOC_CONF"]

        print(f"[Exec] Running StyleID with {unique_name}...")
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=project_root, env=my_env)
        except subprocess.CalledProcessError as e:
            raise Exception(f"Lỗi AI Core: {e.stderr}")

        # 6. TRẢ KẾT QUẢ
        files = [f for f in os.listdir(output_dir) if f.lower().endswith(('.png', '.jpg'))]
        if not files: raise Exception("Không có ảnh output!")
        
        files.sort(key=lambda x: os.path.getmtime(os.path.join(output_dir, x)))
        final_img_path = os.path.join(output_dir, files[-1])
        
        return (torch.from_numpy(np.array(Image.open(final_img_path).convert("RGB")).astype(np.float32) / 255.0)[None,],)

NODE_CLASS_MAPPINGS = { "StyleID_Wrapper": StyleID_Wrapper_Node }
NODE_DISPLAY_NAME_MAPPINGS = { "StyleID_Wrapper": "🇻🇳 Heritage Art StyleID Runner V3" }