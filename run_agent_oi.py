#!/usr/bin/env python
"""
AI Agent for Vietnamese Art Director - Open Interpreter Version
Uses Open Interpreter with local Llama 3.1 via Ollama
This version properly configures the interpreter to execute Python code.
"""

import os
import sys

# CRITICAL FIX: Set environment variables BEFORE any other imports
# This forces litellm to use localhost instead of real OpenAI servers
os.environ["OPENAI_API_BASE"] = "http://localhost:11434/v1"
os.environ["OPENAI_API_KEY"] = "sk-fake-key-for-local-usage"
os.environ["LITELLM_LOG"] = "ERROR"  # Reduce noise

# Now import interpreter after environment is set
from interpreter import interpreter

def setup_interpreter():
    """Configure Open Interpreter for local Llama 3.1 with proper execution"""
    
    # Basic configuration
    interpreter.offline = True
    
    # Use OpenAI-compatible mode pointing to localhost
    interpreter.llm.model = "openai/llama3.1"
    interpreter.llm.api_base = "http://localhost:11434/v1"
    interpreter.llm.api_key = "sk-fake-key-for-local-usage"
    
    interpreter.llm.max_tokens = 4096
    interpreter.llm.context_window = 12000
    
    # CRUCIAL: Enable auto-run to avoid permission prompts
    interpreter.auto_run = True
    
    # FIX: REMOVED interpreter.computer.languages line to prevent AttributeError
    # The default configuration already supports Python correctly
    
    # System message - EXTREMELY STRICT to prevent JSON output
    interpreter.system_message = """Bạn là AI Art Director thực thụ. Bạn ĐANG KẾT NỐI TRỰC TIẾP với hệ thống xử lý ảnh.

QUY TẮC BẤT DI BẤT DỊCH (CRITICAL RULES):
1. CÁC HÀM SAU TỒN TẠI TRONG MODULE `agent_modules`:
   - `get_random_image(category, keyword)` trong agent_modules.data_manager
   - `run_style_transfer(content_path, style_path)` trong agent_modules.styleid_wrapper
   - `insert_object(background_path, object_path)` trong agent_modules.object_insert
   
2. BẮT BUỘC: PHẢI IMPORT các hàm này ở ĐẦU CODE.
3. KHÔNG BAO GIỜ định nghĩa lại các hàm này bằng `def`.
4. KHÔNG BAO GIỜ dùng dữ liệu giả (dummy dictionaries/paths).
5. NHIỆM VỤ DUY NHẤT: Import và GỌI các hàm với tham số thực.

TUYỆT ĐỐI CẤM (ANTI-PATTERNS):
1. TUYỆT ĐỐI CẤM TRẢ VỀ JSON thuần túy (như `{"type": "function", "name": "get_random_image", ...}`).
2. KHÔNG BAO GIỜ chỉ liệt kê tham số hoặc mô tả hàm.
3. KHÔNG BAO GIỜ trả về text giải thích mà không có code.
4. BẮT BUỘC phải đặt code trong khối markdown Python:
   ```python
   # Code thực thi ở đây
   ```

5. BẮT BUỘC: Mỗi câu trả lời PHẢI bắt đầu bằng ```python
6. BẮT BUỘC TRẢ VỀ CODE PYTHON TRONG KHỐI MARKDOWN, KHÔNG PHẢI JSON.

KEYWORDS HỢP LỆ:
- Content: "dinh độc lập", "hồ gươm", "chùa ngọc hoàng", "địa đạo củ chi"
- Style: "đông hồ", "sơn mài", "dân tộc thiểu số", "hàng trống", "khắc gỗ", "lụa", "làng sinh huế", "sơn dầu"
- Object: "nón lá"

VÍ DỤ ĐÚNG (BẮT BUỘC LÀM NHƯ VẬY):
```python
# 1. IMPORT CÁC HÀM (BẮT BUỘC)
from agent_modules.data_manager import get_random_image
from agent_modules.styleid_wrapper import run_style_transfer
from agent_modules.object_insert import insert_object

# 2. Tìm ảnh content và style
content_img = get_random_image('cnt', 'dinh độc lập')
style_img = get_random_image('sty', 'đông hồ')
print(f"Content: {content_img}")
print(f"Style: {style_img}")

# 3. Chạy Style Transfer
if content_img and style_img:
    result = run_style_transfer(content_img, style_img)
    print(f"Stylized: {result}")
    
    # 4. Thêm object nếu cần
    obj_img = get_random_image('obj', 'nón lá')
    if obj_img:
        final = insert_object(result, obj_img)
        print(f"Final: {final}")
```

VÍ DỤ SAI (TUYỆT ĐỐI KHÔNG LÀM):
```json
{
  "type": "function",
  "name": "get_random_image",
  "arguments": {"category": "cnt", "keyword": "dinh độc lập"}
}
```

HOẶC:
```
Tôi sẽ gọi hàm get_random_image với tham số...
```

LƯU Ý:
- LUÔN LUÔN import các hàm từ agent_modules trước khi dùng
- Các hàm sẽ TRẢ VỀ ĐƯỜNG DẪN THẬT từ hệ thống file
- Model StyleID sẽ CHẠY THẬT và tốn vài phút
- LUÔN LUÔN bắt đầu với ```python và kết thúc với ```
- TUYỆT ĐỐI KHÔNG TRẢ VỀ JSON
"""

    # THE FIX: Inject helper functions into interpreter's Python context
    print("[INFO] Initializing Python environment with helper functions...")
    
    # Pre-load the modules so they are available
    setup_code = """
import sys
import os

# Add current directory to path
current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Verify modules are importable
try:
    from agent_modules.data_manager import get_random_image, KEYWORD_MAPPING
    from agent_modules.styleid_wrapper import run_style_transfer
    from agent_modules.object_insert import insert_object
    print("[SUCCESS] Helper functions are available for import!")
    print("   - agent_modules.data_manager.get_random_image")
    print("   - agent_modules.styleid_wrapper.run_style_transfer")
    print("   - agent_modules.object_insert.insert_object")
except ImportError as e:
    print(f"[WARNING] Import error: {e}")
"""
    
    try:
        result = interpreter.computer.run("python", setup_code)
        print("[SUCCESS] Interpreter setup complete!")
    except Exception as e:
        print(f"[WARNING] Warning during setup: {e}")
        print("Continuing anyway...")

def print_welcome():
    """Print welcome message"""
    print("=" * 60)
    print("Vietnamese Art Director AI Agent (Open Interpreter)")
    print("=" * 60)
    print("\nChào mừng! Tôi là trợ lý nghệ thuật AI của bạn.")
    print("\nBạn có thể yêu cầu tôi tạo tác phẩm nghệ thuật bằng cách kết hợp:")
    print("  [Địa điểm] Dinh Độc Lập, Hồ Gươm, Chùa Ngọc Hoàng, Địa Đạo Củ Chi")
    print("  [Phong cách] Đông Hồ, Sơn Mài, Dân Tộc Thiểu Số, Hàng Trống, Khắc Gỗ, Lụa, Làng Sinh Huế, Sơn Dầu")
    print("  [Đối tượng] Nón Lá")
    print("\n[Ví dụ]")
    print("   - 'Vẽ Dinh Độc Lập theo phong cách Đông Hồ'")
    print("   - 'Tạo ảnh Hồ Gươm với phong cách Sơn Mài và thêm nón lá'")
    print("   - 'Hãy tạo tác phẩm về Chùa Ngọc Hoàng theo phong cách Hàng Trống'")
    print("\nGõ 'exit' hoặc 'quit' để thoát.")
    print("=" * 60 + "\n")

def verify_ollama():
    """Verify that Ollama is running"""
    import requests
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            llama_models = [m for m in models if 'llama3.1' in m.get('name', '').lower()]
            if llama_models:
                print(f"[SUCCESS] Ollama is running with Llama 3.1: {llama_models[0]['name']}")
                return True
            else:
                print("[WARNING] Ollama is running but Llama 3.1 not found")
                print("   Please run: ollama pull llama3.1")
                return False
        else:
            print("[WARNING] Ollama API returned unexpected status")
            return False
    except requests.exceptions.RequestException as e:
        print("[ERROR] Cannot connect to Ollama!")
        print("   Please make sure Ollama is running:")
        print("   1. Open a terminal")
        print("   2. Run: ollama serve")
        print(f"   Error: {e}")
        return False

def main():
    """Main function to run the AI agent with Open Interpreter"""
    
    # Check Ollama first
    print("[INFO] Checking Ollama status...")
    if not verify_ollama():
        print("\n[ERROR] Cannot proceed without Ollama. Exiting...")
        return
    
    print()
    
    # Setup interpreter
    setup_interpreter()
    print()
    
    # Print welcome
    print_welcome()
    
    print("[INFO] Agent đã sẵn sàng! Hãy cho tôi biết bạn muốn tạo tác phẩm gì.\n")
    
    # Chat loop
    while True:
        try:
            # Get user input
            user_input = input("Bạn: ").strip()
            
            # Check for exit commands
            if user_input.lower() in ['exit', 'quit', 'thoát', 'thoat']:
                print("\n[INFO] Tạm biệt! Chúc bạn một ngày tốt lành!")
                break
            
            if not user_input:
                continue
            
            # Enhanced prompt with ANTI-JSON instructions
            enhanced_prompt = f"""
YÊU CẦU: "{user_input}"

CẢNH BÁO: TUYỆT ĐỐI KHÔNG TRẢ VỀ JSON. CHỈ TRẢ VỀ PYTHON CODE TRONG KHỐI MARKDOWN.

HÃY VIẾT CODE PYTHON HOÀN CHỈNH (BẮT ĐẦU VỚI ```python):

BƯỚC 1: Phân tích yêu cầu
- Xác định Địa điểm (Content): từ [dinh độc lập, hồ gươm, chùa ngọc hoàng, địa đạo củ chi]
- Xác định Phong cách (Style): từ [đông hồ, sơn mài, dân tộc thiểu số, hàng trống, khắc gỗ, lụa, làng sinh huế, sơn dầu]
- Kiểm tra Đối tượng (Object): Có từ "thêm/với/có" + "nón lá" không?

BƯỚC 2: VIẾT CODE PYTHON THỰC THI (BẮT BUỘC BẮT ĐẦU VỚI ```python):

```python
# 1. BẮT BUỘC: Import các hàm cần thiết
from agent_modules.data_manager import get_random_image
from agent_modules.styleid_wrapper import run_style_transfer
from agent_modules.object_insert import insert_object

# 2. Tìm ảnh Content và Style
# Lưu ý: category phải là 'cnt', 'sty', hoặc 'obj'
content_img = get_random_image('cnt', '<địa_điểm>')  # VD: 'dinh độc lập'
style_img = get_random_image('sty', '<phong_cách>')   # VD: 'đông hồ'

print(f"[Content Path] {{content_img}}")
print(f"[Style Path] {{style_img}}")

# 3. Thực thi StyleID Style Transfer (mất vài phút)
if content_img and style_img:
    print("[INFO] Đang chạy StyleID... (có thể mất 2-3 phút)")
    result = run_style_transfer(content_img, style_img)
    print(f"[Stylized Result] {{result}}")
    
    # 4. Chèn Object (chỉ khi người dùng yêu cầu)
    # Kiểm tra xem có từ "thêm", "với", "có" không
    if <có_thêm_object>:  # True/False tùy theo phân tích
        obj_img = get_random_image('obj', 'nón lá')
        if obj_img and result:
            print("[INFO] Đang thêm đối tượng...")
            final = insert_object(result, obj_img)
            print(f"[Final Result] {{final}}")
        else:
            print("[WARNING] Không tìm thấy object image")
    else:
        print("[INFO] Không có yêu cầu thêm đối tượng")
else:
    print("[ERROR] Lỗi: Không tìm thấy ảnh content hoặc style")
```

LƯU Ý QUAN TRỌNG:
- PHẢI bắt đầu với ```python
- PHẢI có dòng import ở đầu code
- KHÔNG được trả về JSON
- KHÔNG được chỉ giải thích
- KHÔNG được định nghĩa lại các hàm
- PHẢI thực thi code thật

BẮT ĐẦU VIẾT CODE PYTHON NGAY (BẮT ĐẦU VỚI ```python):
"""
            
            # SILENT MODE: Process with interpreter without displaying intermediate steps
            print("\n[INFO] Dang tien hanh tao anh... (Vui long doi 1-2 phut)\n")
            
            response = interpreter.chat(enhanced_prompt, display=False)
            
            # Check output messages to show relevant logs to user
            if response:
                for msg in response:
                    # Only print console outputs (print statements from the generated code)
                    if msg.get('type') == 'console':
                        if 'content' in msg and msg['content']:
                            # Print the content cleanly
                            print(msg['content'].strip())
            
            print("\n" + "=" * 60)
            
        except KeyboardInterrupt:
            print("\n\n[INFO] Đã ngắt kết nối. Tạm biệt!")
            break
        except Exception as e:
            print(f"\n[ERROR] Lỗi: {e}")
            import traceback
            traceback.print_exc()
            print("\nVui lòng thử lại.\n")

if __name__ == "__main__":
    # Verify we're in the right directory
    if not os.path.exists("run_styleid.py"):
        print("[ERROR] Error: Please run this script from the StyleID root directory")
        print(f"   Current directory: {os.getcwd()}")
        print("   Expected files: run_styleid.py, agent_modules/")
        sys.exit(1)
    
    # Verify agent_modules exists
    if not os.path.exists("agent_modules"):
        print("[ERROR] Error: agent_modules folder not found!")
        print("   Please ensure you have created the agent_modules package")
        sys.exit(1)
    
    main()