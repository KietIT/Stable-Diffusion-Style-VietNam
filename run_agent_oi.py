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
1. CÁC HÀM SAU TỒN TẠI TRONG MODULE `agent_modules` VỚI CHỮ KÝ CỐ ĐỊNH:
   - `get_random_image(category, keyword)` - CHỈ 2 THAM SỐ!
     Tham số 1: 'cnt', 'sty', hoặc 'obj'
     Tham số 2: Keyword (VD: 'Dinh Độc Lập', 'son_dau', 'non_la')
   - `run_style_transfer(content_path, style_path)` - CHỈ 2 THAM SỐ!
   - `insert_object(background_path, object_path)` - CHỈ 2 THAM SỐ!
   
2. BẮT BUỘC: PHẢI IMPORT các hàm này ở ĐẦU CODE.
3. KHÔNG BAO GIỜ định nghĩa lại các hàm này bằng `def`.
4. KHÔNG BAO GIỜ dùng dữ liệu giả (dummy dictionaries/paths).
5. KHÔNG BAO GIỜ gọi hàm với số tham số SAI (VD: 3 tham số thay vì 2).
6. NHIỆM VỤ DUY NHẤT: Import và GỌI các hàm với ĐÚNG SỐ THAM SỐ.

TUYỆT ĐỐI CẤM (ANTI-PATTERNS):
1. TUYỆT ĐỐI CẤM TRẢ VỀ JSON thuần túy.
2. KHÔNG BAO GIỜ chỉ liệt kê tham số hoặc mô tả hàm.
3. KHÔNG BAO GIỜ trả về text giải thích mà không có code.
4. BẮT BUỘC phải đặt code trong khối markdown Python.
5. BẮT BUỘC: Mỗi câu trả lời PHẢI bắt đầu bằng ```python
6. BẮT BUỘC TRẢ VỀ CODE PYTHON TRONG KHỐI MARKDOWN, KHÔNG PHẢI JSON.

DANH SÁCH CỐ ĐỊNH:
- 8 Phong cách: đông hồ, sơn mài, sơn dầu, làng sinh huế, lụa, khắc gỗ, hàng trống, dân tộc thiểu số
- 185 Di tích lịch sử (trích xuất tên từ yêu cầu)
- 1 Đối tượng: nón lá

VÍ DỤ ĐÚNG (BẮT BUỘC LÀM NHƯ VẬY):
```python
# 1. IMPORT CÁC HÀM (BẮT BUỘC)
from agent_modules.data_manager import get_random_image
from agent_modules.styleid_wrapper import run_style_transfer
from agent_modules.object_insert import insert_object

# 2. Tìm ảnh content và style
content_img = get_random_image('cnt', 'Dinh Độc Lập')
style_img = get_random_image('sty', 'dong_ho')
print(f"Content: {content_img}")
print(f"Style: {style_img}")

# 3. Chạy Style Transfer
if content_img and style_img:
    result = run_style_transfer(content_img, style_img)
    print(f"Stylized: {result}")
    
    # 4. Thêm object nếu cần
    if 'nón' in user_request.lower():
        obj_img = get_random_image('obj', 'non_la')
        if obj_img:
            final = insert_object(result, obj_img)
            print(f"Final: {final}")
```

VÍ DỤ SAI (TUYỆT ĐỐI KHÔNG LÀM):
```json
{
  "type": "function",
  "name": "get_random_image"
}
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
    print("  [Địa điểm] 185 Di tích lịch sử Việt Nam")
    print("  [Phong cách] Đông Hồ, Sơn Mài, Sơn Dầu, Làng Sinh Huế, Lụa, Khắc Gỗ, Hàng Trống, Dân Tộc Thiểu Số")
    print("  [Đối tượng] Nón Lá")
    print("\n[Ví dụ]")
    print("   - 'Vẽ Dinh Độc Lập theo phong cách Đông Hồ'")
    print("   - 'Tạo ảnh Cố Đô Huế với phong cách Sơn Mài và thêm nón lá'")
    print("   - 'Hãy tạo tác phẩm về Chùa Hương theo phong cách Hàng Trống'")
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
            
            # Enhanced prompt with STRICT DATASET CONSTRAINTS
            enhanced_prompt = f"""
YÊU CẦU: "{user_input}"

BẠN LÀ CHUYÊN GIA TRÍCH XUẤT TỪ KHÓA. NHIỆM VỤ LÀ MAPPING YÊU CẦU VÀO DANH SÁCH CÓ SẴN.

1. BẢNG MAPPING PHONG CÁCH (STYLE) - CHỈ ĐƯỢC CHỌN 1 TRONG 8 KEY SAU:
   - "đông hồ" -> 'dong_ho'
   - "sơn mài" -> 'son_mai'
   - "sơn dầu" -> 'son_dau'
   - "làng sinh", "làng sinh huế" -> 'lang_sinh'
   - "lụa", "tranh lụa" -> 'tranh_lua'
   - "khắc gỗ" -> 'khac_go'
   - "hàng trống" -> 'hang_trong'
   - "dân tộc thiểu số", "người mèo", "thổ cẩm" -> 'dan_toc_thieu_so'

2. ĐỐI TƯỢNG (OBJECT) - HỆ THỐNG TỰ ĐỘNG PHÁT HIỆN:
   - Hiện có: "nón lá", "non la", "nón" -> 'non_la'
   - Có thể thêm: "áo dài", "đèn lồng", "quạt giấy"...
   - (Nếu không nhắc đến đối tượng -> Bỏ qua)

3. ĐỊA ĐIỂM (CONTENT) - TRONG DANH SÁCH 185 DI TÍCH:
   - Hãy trích xuất TÊN DI TÍCH hoặc ĐỊA DANH trong câu.
   - Ví dụ: "Dinh Độc Lập", "Cố Đô Huế", "Chùa Hương", "Vịnh Hạ Long"...
   - Giữ nguyên tiếng Việt có dấu để tìm kiếm trong Database.

HÃY VIẾT CODE PYTHON TRONG KHỐI MARKDOWN ĐỂ THỰC THI:
(Lưu ý: Không giải thích, không dùng JSON, chỉ viết Code)

QUAN TRỌNG - CHỮ KÝ HÀM (FUNCTION SIGNATURE):
- get_random_image(category, keyword_vn) - CHỈ 2 THAM SỐ
  Ví dụ: get_random_image('cnt', 'Dinh Độc Lập')
  Ví dụ: get_random_image('sty', 'son_dau')
  SAI: get_random_image('sty', 'Dinh Độc Lập', 'son_dau') ← TUYỆT ĐỐI KHÔNG!

```python
# 1. Import
from agent_modules.data_manager import get_random_image
from agent_modules.styleid_wrapper import run_style_transfer
from agent_modules.object_insert import insert_object

# 2. Định nghĩa Key sau khi phân tích
# User request: "{user_input}"
style_key = '...'    # Điền key từ bảng Mapping trên (VD: 'son_dau', 'hang_trong')
content_name = '...' # Điền tên di tích trích xuất được (VD: 'Chùa Một Cột')

# 3. Tìm ảnh - CHÚ Ý: Mỗi hàm CHỈ 2 THAM SỐ (category, keyword)
print(f"[INFO] Đang tìm ảnh cho: {{content_name}} theo phong cách {{style_key}}")
content_img = get_random_image('cnt', content_name)  # 2 tham số: category='cnt', keyword=content_name
style_img = get_random_image('sty', style_key)       # 2 tham số: category='sty', keyword=style_key

if not content_img:
    print(f"[ERROR] Không tìm thấy ảnh di tích: {{content_name}}")
elif not style_img:
    print(f"[ERROR] Không tìm thấy ảnh phong cách: {{style_key}}")
else:
    print(f"[Content Path] {{content_img}}")
    print(f"[Style Path] {{style_img}}")

    # 4. Chạy Style Transfer
    result = run_style_transfer(content_img, style_img)
    print(f"[Stylized Result] {{result}}")

    # 5. Chèn Object (Tự động phát hiện từ user input)
    from agent_modules.data_manager import detect_object_in_text
    
    detected_object = detect_object_in_text('{user_input}')
    if detected_object:
        obj_img = get_random_image('obj', detected_object)
        if obj_img:
            print(f"[Object Path] {{obj_img}}")
            final = insert_object(result, obj_img)
            if final:
                print(f"[Final Result] {{final}}")
            else:
                print(f"[WARNING] Object insertion failed, using stylized result: {{result}}")
        else:
            print(f"[WARNING] Object folder not found for: {{detected_object}}")
```

LƯU Ý QUAN TRỌNG:
- PHẢI bắt đầu với ```python
- PHẢI có dòng import ở đầu code
- PHẢI điền đúng style_key từ 8 giá trị mapping
- PHẢI điền content_name là tên di tích tiếng Việt có dấu
- KHÔNG được trả về JSON
- KHÔNG được chỉ giải thích
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