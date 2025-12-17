import os
import subprocess
import torch
import numpy as np
from PIL import Image
import folder_paths
import time  
import shutil
import random
# Danh sách Style của bạn
STYLE_LIST = ["son_dau", "son_mai", "dong_ho", "hang_trong", "khac_go", "lua", "lang_sinh_hue", "dan_toc_thieu_so"]

class StyleID_Wrapper_Node:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        default_python = "/opt/conda/envs/styleid_env/bin/python" 
        default_root = "/workspace/Stable-Diffusion-Style-VietNam"
        if os.name == 'nt': 
            default_python = "D:/Anaconda/envs/StyleID/python.exe"
            default_root = "D:/Job_Parttime/StyleID"

        return {
            "required": {
                "content_image": ("IMAGE",), 
                "style_name": (STYLE_LIST,),
                "python_path": ("STRING", {"default": default_python}),
                "project_root": ("STRING", {"default": default_root}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run_styleid"
    CATEGORY = "HeritageArt"

    def run_styleid(self, content_image, style_name, python_path, project_root):
        torch.cuda.empty_cache()
        
        # 1. SETUP ĐƯỜNG DẪN CƠ BẢN
        if python_path == "auto" or not python_path: pass 
        project_root = os.path.abspath(project_root)
        script_path = os.path.join(project_root, "run_styleid.py")
        config_path = os.path.join(project_root, "models", "ldm", "stable-diffusion-v1", "v1-inference.yaml")
        ckpt_path = os.path.join(project_root, "models", "ldm", "stable-diffusion-v1", "model.ckpt")
        data_dir = os.path.join(project_root, "data")
        
        # Đường dẫn folder style gốc (chứa nhiều ảnh)
        sty_dir_orig = os.path.join(data_dir, "sty", style_name)

        # 2. XỬ LÝ LOGIC RANDOM STYLE (QUAN TRỌNG)
        # Tạo folder style tạm thời chỉ chứa ĐÚNG 1 ẢNH
        temp_sty_dir = os.path.join(project_root, "temp_style_input")
        if os.path.exists(temp_sty_dir):
            shutil.rmtree(temp_sty_dir) # Xóa folder cũ
            time.sleep(0.1)
        os.makedirs(temp_sty_dir, exist_ok=True)

        try:
            # Lấy danh sách ảnh trong folder gốc
            valid_ext = ('.png', '.jpg', '.jpeg')
            all_style_images = [f for f in os.listdir(sty_dir_orig) if f.lower().endswith(valid_ext)]
            
            if not all_style_images:
                raise Exception(f"Không tìm thấy ảnh nào trong folder style: {sty_dir_orig}")
            
            # Chọn ngẫu nhiên 1 ảnh
            chosen_style_img = random.choice(all_style_images)
            print(f"[Random] Đã chọn style: {chosen_style_img}")
            
            # Copy ảnh đó vào folder tạm
            shutil.copy2(os.path.join(sty_dir_orig, chosen_style_img), 
                         os.path.join(temp_sty_dir, chosen_style_img))
            
        except Exception as e:
            raise Exception(f"Lỗi khi chọn Random Style: {str(e)}")

        # 3. CHUẨN BỊ ẢNH INPUT (Từ ComfyUI)
        temp_input_dir = os.path.join(project_root, "temp_user_input")
        os.makedirs(temp_input_dir, exist_ok=True)
        
        img_tensor = content_image[0].cpu().numpy() * 255.0
        img_tensor = np.clip(img_tensor, 0, 255).astype(np.uint8)
        input_pil = Image.fromarray(img_tensor)
        temp_img_path = os.path.join(temp_input_dir, "input_base.png")
        input_pil.save(temp_img_path)

        # 4. CHUẨN BỊ FOLDER OUTPUT & DỌN DẸP CŨ
        output_dir = os.path.join(project_root, "temp_output")
        if os.path.exists(output_dir):
            try:
                shutil.rmtree(output_dir)
                time.sleep(0.5)
            except: pass
        os.makedirs(output_dir, exist_ok=True)

        # 5. GỌI SUBPROCESS
        # Lưu ý: --sty trỏ vào temp_sty_dir (chỉ chứa 1 ảnh)
        cmd = [
            python_path, script_path,
            "--cnt", temp_input_dir,
            "--sty", temp_sty_dir, 
            "--output_path", output_dir,
            "--model_config", config_path,
            "--ckpt", ckpt_path,
            "--low_vram_mode"
        ]

        my_env = os.environ.copy()
        if "PYTORCH_CUDA_ALLOC_CONF" in my_env: del my_env["PYTORCH_CUDA_ALLOC_CONF"]

        print(f"[Exec] Running StyleID...")
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=project_root, env=my_env)
        except subprocess.CalledProcessError as e:
            if os.path.exists(temp_img_path): os.remove(temp_img_path)
            raise Exception(f"StyleID Failed: {e.stderr}")

        # 6. LỌC VÀ LẤY ẢNH KẾT QUẢ CUỐI CÙNG (CLEANUP OUTPUT)
        try:
            # Lấy tất cả file ảnh
            files = [f for f in os.listdir(output_dir) if f.lower().endswith(('.png', '.jpg'))]
            if not files: raise Exception("Không có ảnh output nào được tạo ra!")

            # Logic lọc: File tạo sau cùng là file kết quả (bỏ qua các file _0_, _10_ cũ hơn)
            # Sắp xếp theo thời gian tạo (Mới nhất nằm cuối)
            files.sort(key=lambda x: os.path.getmtime(os.path.join(output_dir, x)))
            latest_file_name = files[-1] # Lấy file cuối cùng
            latest_file_path = os.path.join(output_dir, latest_file_name)

            print(f"[ComfyUI] Kết quả cuối cùng: {latest_file_name}")

            # Xóa các file thừa (như file _0_mask.png)
            for f in files[:-1]:
                try:
                    os.remove(os.path.join(output_dir, f))
                    print(f"[Cleanup] Đã xóa file tạm: {f}")
                except: pass

            # Load ảnh cuối cùng
            img = None
            for i in range(3):
                try:
                    img = Image.open(latest_file_path)
                    img.load()
                    break
                except: time.sleep(1)
            
            if img is None: raise Exception("Không thể mở file ảnh output")

            img = img.convert("RGB")
            img = np.array(img).astype(np.float32) / 255.0
            img = torch.from_numpy(img)[None,]
            
            # Dọn dẹp input temp
            if os.path.exists(temp_img_path): os.remove(temp_img_path)
            
            return (img,)
            
        except Exception as e:
            raise Exception(f"Output Load Error: {str(e)}")

NODE_CLASS_MAPPINGS = { "StyleID_Wrapper": StyleID_Wrapper_Node }
NODE_DISPLAY_NAME_MAPPINGS = { "StyleID_Wrapper": "🇻🇳 Heritage Art StyleID Runner" }