#!/usr/bin/env python
"""
AI Agent for Vietnamese Art Director
Uses Open Interpreter with local Llama 3.1 via Ollama
"""

import os
import sys
import re

# Import agent modules
from agent_modules.data_manager import get_random_image, get_available_keywords, KEYWORD_MAPPING
from agent_modules.styleid_wrapper import run_style_transfer
from agent_modules.object_insert import insert_object

def print_welcome():
    """Print welcome message"""
    print("=" * 60)
    print("🎨 Vietnamese Art Director AI Agent")
    print("=" * 60)
    print("\nChào mừng! Tôi là trợ lý nghệ thuật của bạn.")
    print("\nBạn có thể yêu cầu tôi tạo tác phẩm nghệ thuật bằng cách kết hợp:")
    print("  📍 Địa điểm: Dinh Độc Lập, Hồ Gươm, Chùa Ngọc Hoàng, Địa Đạo Củ Chi")
    print("  🎨 Phong cách: Đông Hồ, Sơn Mài, Dân Tộc Thiểu Số, Hàng Trống, Khắc Gỗ, Lụa, Làng Sinh Huế, Sơn Dầu")
    print("  🎯 Đối tượng: Nón Lá")
    print("\nVí dụ: 'Vẽ Dinh Độc Lập theo phong cách Đông Hồ và thêm nón lá'")
    print("\nGõ 'exit' hoặc 'quit' để thoát.")
    print("=" * 60 + "\n")

def extract_keywords(user_input):
    """
    Extract location, style, and object keywords from user input
    """
    user_lower = user_input.lower()
    
    # Define all available keywords
    locations = {
        "dinh độc lập": "dinh_doc_lap",
        "dinh doc lap": "dinh_doc_lap",
        "hồ gươm": "ho_guom",
        "ho guom": "ho_guom",
        "chùa ngọc hoàng": "chua_ngoc_hoang",
        "chua ngoc hoang": "chua_ngoc_hoang",
        "địa đạo củ chi": "dia_dao_cu_chi",
        "dia dao cu chi": "dia_dao_cu_chi"
    }
    
    styles = {
        "đông hồ": "dong_ho",
        "dong ho": "dong_ho",
        "sơn mài": "son_mai",
        "son mai": "son_mai",
        "dân tộc thiểu số": "dan_toc_thieu_so",
        "dan toc thieu so": "dan_toc_thieu_so",
        "hàng trống": "hang_trong",
        "hang trong": "hang_trong",
        "khắc gỗ": "khac_go",
        "khac go": "khac_go",
        "lụa": "lua",
        "lua": "lua",
        "làng sinh huế": "lang_sinh_hue",
        "lang sinh hue": "lang_sinh_hue",
        "sơn dầu": "son_dau",
        "son dau": "son_dau"
    }
    
    objects = {
        "nón lá": "non_la",
        "non la": "non_la"
    }
    
    # Extract keywords
    location = None
    style = None
    obj = None
    
    for key, value in locations.items():
        if key in user_lower:
            location = key
            break
    
    for key, value in styles.items():
        if key in user_lower:
            style = key
            break
    
    for key, value in objects.items():
        if key in user_lower:
            obj = key
            break
    
    return location, style, obj

def execute_workflow(location_kw, style_kw, object_kw=None):
    """
    Complete workflow for Vietnamese art creation
    """
    print(f"\n🎯 Bắt đầu quy trình tạo tác phẩm...")
    print(f"   📍 Địa điểm: {location_kw}")
    print(f"   🎨 Phong cách: {style_kw}")
    if object_kw:
        print(f"   🎯 Đối tượng: {object_kw}")
    print()
    
    # Step 1: Get content image
    print("1️⃣ Lấy ảnh nội dung...")
    content_path = get_random_image('cnt', location_kw)
    if not content_path:
        print("❌ Không tìm thấy ảnh nội dung!")
        return None
    print(f"   ✅ {content_path}")
    
    # Step 2: Get style image
    print("\n2️⃣ Lấy ảnh phong cách...")
    style_path = get_random_image('sty', style_kw)
    if not style_path:
        print("❌ Không tìm thấy ảnh phong cách!")
        return None
    print(f"   ✅ {style_path}")
    
    # Step 3: Apply style transfer
    print("\n3️⃣ Áp dụng chuyển đổi phong cách...")
    print("   ⏳ Quá trình này có thể mất vài phút, vui lòng đợi...")
    stylized_path = run_style_transfer(content_path, style_path)
    if not stylized_path:
        print("❌ Lỗi khi áp dụng phong cách!")
        return None
    print(f"   ✅ Hoàn thành: {stylized_path}")
    
    # Step 4: Insert object if requested
    if object_kw:
        print("\n4️⃣ Thêm đối tượng...")
        object_path = get_random_image('obj', object_kw)
        if not object_path:
            print("⚠️ Không tìm thấy ảnh đối tượng, bỏ qua bước này")
            return stylized_path
        
        print(f"   📦 Đối tượng: {object_path}")
        final_path = insert_object(stylized_path, object_path)
        if final_path:
            print(f"   ✅ Hoàn thành: {final_path}")
            return final_path
        else:
            print("⚠️ Lỗi khi thêm đối tượng, trả về ảnh đã stylize")
            return stylized_path
    
    return stylized_path

def process_request(user_input):
    """
    Process user request and execute workflow
    """
    # Extract keywords from user input
    location, style, obj = extract_keywords(user_input)
    
    # Validate input
    if not location:
        print("❌ Không tìm thấy địa điểm trong yêu cầu!")
        return None
    
    if not style:
        print("❌ Không tìm thấy phong cách trong yêu cầu!")
        return None
    
    print(f"\n📝 Đã nhận diện:")
    print(f"   Địa điểm: {location}")
    print(f"   Phong cách: {style}")
    if obj:
        print(f"   Đối tượng: {obj}")
    
    # Execute workflow
    result = execute_workflow(location, style, obj)
    
    if result:
        print(f"\n✨ Hoàn thành! Tác phẩm của bạn đã được lưu tại:")
        print(f"   📁 {result}")
        return result
    else:
        print("\n❌ Có lỗi xảy ra trong quá trình tạo tác phẩm.")
        return None

def main():
    """Main function to run the AI agent"""
    print_welcome()
    
    print("🤖 Agent đã sẵn sàng! Bạn muốn tạo tác phẩm gì?\n")
    
    # Chat loop
    while True:
        try:
            # Get user input
            user_input = input("Bạn: ").strip()
            
            # Check for exit commands
            if user_input.lower() in ['exit', 'quit', 'thoát', 'thoat']:
                print("\n👋 Tạm biệt! Chúc bạn một ngày tốt lành!")
                break
            
            if not user_input:
                continue
            
            # Process request
            print()
            process_request(user_input)
            print()
            
        except KeyboardInterrupt:
            print("\n\n👋 Đã ngắt kết nối. Tạm biệt!")
            break
        except Exception as e:
            print(f"\n❌ Lỗi: {e}")
            import traceback
            traceback.print_exc()
            print("Vui lòng thử lại.\n")

if __name__ == "__main__":
    main()