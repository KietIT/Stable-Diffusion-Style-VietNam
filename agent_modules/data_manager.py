import os
import random
from typing import Optional

# Object keywords mapping (extensible for future objects)
OBJECT_KEYWORDS = {
    "nón lá": "non_la",
    "non la": "non_la",
    "non_la": "non_la",
    "nón": "non_la",
    "non": "non_la",
    # Add more objects here in the future:
    # "áo dài": "ao_dai",
    # "đèn lồng": "den_long",
    # "quạt giấy": "quat_giay",
}

KEYWORD_MAPPING = {
    # Content (Locations)
    "dinh độc lập": "dinh_doc_lap",
    "chùa ngọc hoàng": "chua_ngoc_hoang",
    "địa đạo củ chi": "dia_dao_cu_chi",
    # Style - Vietnamese with diacritics
    "đông hồ": "dong_ho",
    "sơn mài": "son_mai",
    "sơn dầu": "son_dau",
    "dân tộc thiểu số": "dan_toc_thieu_so",
    "hàng trống": "hang_trong",
    "khắc gỗ": "khac_go",
    "lụa": "lua",
    "làng sinh huế": "lang_sinh_hue",
    # Style - English keys for LLM compatibility
    "dong_ho": "dong_ho",
    "son_mai": "son_mai",
    "son_dau": "son_dau",
    "dan_toc_thieu_so": "dan_toc_thieu_so",
    "hang_trong": "hang_trong",
    "khac_go": "khac_go",
    "tranh_lua": "lua",  # Alias: tranh_lua -> lua folder
    "lang_sinh": "lang_sinh_hue",  # Alias: lang_sinh -> lang_sinh_hue folder
}

# Merge object keywords into main mapping
KEYWORD_MAPPING.update(OBJECT_KEYWORDS)

def normalize_vietnamese(text: str) -> str:
    """
    Normalize Vietnamese text by removing diacritics for fuzzy matching.
    Handles special Vietnamese characters like Đ/đ.
    
    Args:
        text: Vietnamese text with or without diacritics
        
    Returns:
        Normalized text without diacritics in lowercase
    """
    import unicodedata
    
    # First, handle special Vietnamese characters that don't decompose with NFD
    vietnamese_chars = {
        'Đ': 'D', 'đ': 'd',
        'Ð': 'D', 'ð': 'd'  # Alternative forms
    }
    
    normalized = text
    for viet_char, replacement in vietnamese_chars.items():
        normalized = normalized.replace(viet_char, replacement)
    
    # Decompose unicode (separate base characters from diacritics)
    nfd = unicodedata.normalize('NFD', normalized)
    # Filter out combining marks (diacritics)
    without_diacritics = ''.join(c for c in nfd if unicodedata.category(c) != 'Mn')
    return without_diacritics.lower().strip()


def get_random_image(category: str, keyword_vn: str) -> Optional[str]:
    """
    Get a random image path based on category and Vietnamese keyword.
    Handles case-insensitive matching and fuzzy matching without diacritics.
    
    Args:
        category: One of 'cnt' (content), 'sty' (style), or 'obj' (object)
        keyword_vn: Vietnamese keyword (e.g., "dinh độc lập", "Dinh Doc Lap", "DINH DOC LAP")
    
    Returns:
        Absolute path to a random image, or None if folder doesn't exist
    """
    # Normalize keyword to lowercase for matching
    keyword_lower = keyword_vn.lower().strip()
    
    # Try exact match first (case-insensitive)
    folder_name = KEYWORD_MAPPING.get(keyword_lower)
    
    # If no exact match, try fuzzy matching without diacritics
    if folder_name is None:
        normalized_input = normalize_vietnamese(keyword_vn)
        
        # Search through all keys with normalized comparison
        for key, value in KEYWORD_MAPPING.items():
            normalized_key = normalize_vietnamese(key)
            if normalized_input == normalized_key:
                folder_name = value
                print(f"[FUZZY MATCH] '{keyword_vn}' → '{key}' → folder '{value}'")
                break
    
    if folder_name is None:
        print(f"Warning: No mapping found for keyword '{keyword_vn}'")
        return None
    
    # Construct folder path
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    folder_path = os.path.join(base_dir, "data", category, folder_name)
    
    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"Error: Folder does not exist: {folder_path}")
        return None
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    image_files = [
        f for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f)) and
        os.path.splitext(f)[1] in image_extensions
    ]
    
    if not image_files:
        print(f"Error: No images found in {folder_path}")
        return None
    
    # Select random image
    selected_image = random.choice(image_files)
    image_path = os.path.abspath(os.path.join(folder_path, selected_image))
    
    print(f"Selected image: {image_path}")
    return image_path


def detect_object_in_text(text: str) -> Optional[str]:
    """
    Detect object keyword in text input.
    Returns the folder name of the detected object, or None if not found.
    
    Args:
        text: User input text to search for object keywords
        
    Returns:
        Folder name of detected object (e.g., 'non_la') or None
    """
    text_lower = text.lower().strip()
    text_normalized = normalize_vietnamese(text)
    
    # Sort by length (longest first) to match "nón lá" before "nón"
    sorted_keywords = sorted(OBJECT_KEYWORDS.keys(), key=len, reverse=True)
    
    for keyword in sorted_keywords:
        keyword_lower = keyword.lower()
        keyword_normalized = normalize_vietnamese(keyword)
        
        # Try exact match first
        if keyword_lower in text_lower:
            folder_name = OBJECT_KEYWORDS[keyword]
            print(f"[OBJECT DETECTED] '{keyword}' → folder '{folder_name}'")
            return folder_name
        
        # Try normalized match (without diacritics)
        if keyword_normalized in text_normalized:
            folder_name = OBJECT_KEYWORDS[keyword]
            print(f"[OBJECT DETECTED - FUZZY] '{keyword}' → folder '{folder_name}'")
            return folder_name
    
    return None


def get_available_objects() -> list:
    """
    Get list of all available object keywords.
    
    Returns:
        List of unique object folder names
    """
    return list(set(OBJECT_KEYWORDS.values()))


def get_available_keywords() -> dict:
    """
    Get all available keywords organized by category.
    
    Returns:
        Dictionary mapping categories to lists of available keywords
    """
    return {
        "locations": ["dinh độc lập", "chùa ngọc hoàng", "địa đạo củ chi"],
        "styles": ["đông hồ", "sơn mài", "dân tộc thiểu số", "hàng trống", "khắc gỗ", "lụa", "làng sinh huế", "sơn dầu"],
        "objects": list(set([k for k in OBJECT_KEYWORDS.keys() if ' ' in k]))  # Return multi-word names only
    }