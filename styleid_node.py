import requests
import torch
import numpy as np
from PIL import Image
import io

class StyleID_Fast_Node:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "content_image": ("IMAGE",), 
                "style_name": (["son_dau", "son_mai", "dong_ho", "hang_trong", "dan_toc_thieu_so", "lua", "khac_go"],),
                "server_url": ("STRING", {"default": "http://127.0.0.1:8000/transform"}),
                "start_step": ("INT", {"default": 34}), 
                "gamma": ("FLOAT", {"default": 0.75}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "send_request"
    CATEGORY = "HeritageArt"

    def send_request(self, content_image, style_name, server_url, start_step, gamma):
        # 1. Chuyển ảnh sang Bytes
        img_np = (content_image[0].cpu().numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)
        buffer = io.BytesIO()
        pil_img.save(buffer, format="PNG")
        buffer.seek(0)

        files = {"file": ("content.png", buffer, "image/png")}
        data = {
            "style_name": style_name,
            "start_step": start_step,
            "gamma": gamma,
            "T": 1.5 
        }
        
        try:
            # Tăng timeout lên 600s
            response = requests.post(server_url, files=files, data=data, timeout=600)
            
            if response.status_code == 200:
                try:
                    output_img = Image.open(io.BytesIO(response.content)).convert("RGB")
                    out_tensor = torch.from_numpy(np.array(output_img).astype(np.float32) / 255.0)[None,]
                    return (out_tensor,)
                except Exception as img_err:
                    # NẾU LỖI NÀY XUẤT HIỆN: Nghĩa là server trả về 200 OK nhưng nội dung không phải ảnh
                    print(f"❌ DATA ERROR: Server returned non-image data!")
                    print(f"   Content preview: {response.content[:200]}") # In 200 ký tự đầu để xem lỗi là gì
                    return (content_image,)
            else:
                print(f"❌ Server returned error {response.status_code}: {response.text}")
                return (content_image,)
        
        except Exception as e:
            print(f"❌ Connection Error: {e}")
            return (content_image,)

# MAPPING CHO NODE
NODE_CLASS_MAPPINGS = {
    "StyleID_Wrapper": StyleID_Fast_Node
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StyleID_Wrapper": "🇻🇳 Heritage Art StyleID (API)"
}