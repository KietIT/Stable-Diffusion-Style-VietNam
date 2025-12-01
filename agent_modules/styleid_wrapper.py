import subprocess
import os
import shutil
from typing import Optional

STYLEID_PYTHON_PATH = r"D:\Anaconda\envs\StyleID\python.exe"

def run_style_transfer(content_path: str, style_path: str, output_dir: str = "output") -> Optional[str]:
    """
    Thực thi StyleID style transfer bằng cách gọi Python của môi trường StyleID (Python 3.8).
    """
    
    # 1. Định vị thư mục gốc (nơi chứa run_styleid.py, giả định file này nằm 1 cấp trên agent_modules)
    # Lấy đường dẫn tuyệt đối của folder chứa run_styleid.py
    # Giả định styleid_wrapper.py nằm trong agent_modules/
    root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    run_styleid_script = os.path.join(root_dir, "run_styleid.py")
    
    # --- KIỂM TRA LỖI BAN ĐẦU ---
    if not os.path.exists(STYLEID_PYTHON_PATH):
        print(f"Error: Python của môi trường StyleID không tìm thấy tại: {STYLEID_PYTHON_PATH}")
        return None
    if not os.path.exists(run_styleid_script):
        print(f"Error: run_styleid.py không tìm thấy tại: {run_styleid_script}")
        return None
    if not os.path.exists(content_path):
        print(f"Error: Ảnh Content không tìm thấy: {content_path}")
        return None
    if not os.path.exists(style_path):
        print(f"Error: Ảnh Style không tìm thấy: {style_path}")
        return None
    
    # --- CHUẨN BỊ THƯ MỤC TẠM (VÌ run_styleid.py GỐC CẦN INPUT LÀ THƯ MỤC) ---
    temp_cnt_dir = os.path.join(root_dir, "temp_cnt")
    temp_sty_dir = os.path.join(root_dir, "temp_sty")
    
    # Đảm bảo thư mục tạm sạch sẽ trước khi tạo
    if os.path.exists(temp_cnt_dir):
        shutil.rmtree(temp_cnt_dir)
    if os.path.exists(temp_sty_dir):
        shutil.rmtree(temp_sty_dir)
        
    os.makedirs(temp_cnt_dir, exist_ok=True)
    os.makedirs(temp_sty_dir, exist_ok=True)
    
    # Copy ảnh vào thư mục tạm
    content_filename = os.path.basename(content_path)
    style_filename = os.path.basename(style_path)
    shutil.copy(content_path, os.path.join(temp_cnt_dir, content_filename))
    shutil.copy(style_path, os.path.join(temp_sty_dir, style_filename))
    
    # --- THIẾT LẬP LỆNH THỰC THI ---
    output_path = os.path.join(root_dir, output_dir)
    os.makedirs(output_path, exist_ok=True)
    
    cmd = [
        STYLEID_PYTHON_PATH, # ⭐ Dùng Python của StyleID (đã sửa)
        run_styleid_script,
        "--cnt", temp_cnt_dir,
        "--sty", temp_sty_dir,
        "--output_path", output_path,
        # Các tham số mặc định của StyleID
        "--ddim_inv_steps", "50",
        "--save_feat_steps", "50",
        "--start_step", "49",
        "--gamma", "0.75",
        "--T", "1.5"
    ]
    
    print(f"Executing StyleID style transfer via StyleID Env...")
    print(f"Command: {' '.join(cmd)}")
    
    # --- CHẠY LỆNH ---
    try:
        # Chạy lệnh. cwd=root_dir là quan trọng để script run_styleid.py tìm đúng các folder con của nó.
        result = subprocess.run(
            cmd,
            cwd=root_dir,
            capture_output=True,
            text=True,
            check=True
        )
        
        print("StyleID execution completed successfully!")
        if result.stdout:
            # print("Output:", result.stdout) # Có thể quá dài, nên comment
            pass
        
        # --- DỌN DẸP VÀ TRẢ VỀ KẾT QUẢ ---
        content_base = os.path.splitext(content_filename)[0]
        style_base = os.path.splitext(style_filename)[0]
        # Định dạng output: {content_name}_stylized_{style_name}.png
        expected_output = f"{content_base}_stylized_{style_base}.png"
        output_image_path = os.path.join(output_path, expected_output)
        
        # Dọn dẹp thư mục tạm dù thành công hay thất bại
        shutil.rmtree(temp_cnt_dir)
        shutil.rmtree(temp_sty_dir)
        
        if os.path.exists(output_image_path):
            print(f"Stylized image saved to: {output_image_path}")
            return output_image_path
        else:
            print(f"Warning: Expected output not found at {output_image_path}. Looking for latest output file...")
            # Logic tìm file mới nhất (đã giữ nguyên logic cũ của anh)
            output_files = [f for f in os.listdir(output_path) if f.endswith('.png')]
            if output_files:
                latest_file = max(
                    [os.path.join(output_path, f) for f in output_files],
                    key=os.path.getctime
                )
                print(f"Found output image: {latest_file}")
                return latest_file
            
            return None
            
    except subprocess.CalledProcessError as e:
        print(f"Error executing StyleID: {e}")
        print(f"StyleID Error Output: {e.stderr}")
        
        # Dọn dẹp cả khi có lỗi
        if os.path.exists(temp_cnt_dir):
            shutil.rmtree(temp_cnt_dir)
        if os.path.exists(temp_sty_dir):
            shutil.rmtree(temp_sty_dir)
            
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None