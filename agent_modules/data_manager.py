import os
import random
from typing import Optional

KEYWORD_MAPPING = {
    "dinh độc lập": "dinh_doc_lap",
    "hồ gươm": "ho_guom",
    "đông hồ": "dong_ho",
    "sơn mài": "son_mai",
    "nón lá": "non_la"
}

def get_random_image(category: str, keyword_vn: str) -> Optional[str]:
    """
    Get a random image path based on category and Vietnamese keyword.
    
    Args:
        category: One of 'cnt' (content), 'sty' (style), or 'obj' (object)
        keyword_vn: Vietnamese keyword (e.g., "dinh độc lập")
    
    Returns:
        Absolute path to a random image, or None if folder doesn't exist
    """
    # Normalize keyword to lowercase for matching
    keyword_lower = keyword_vn.lower().strip()
    
    # Map Vietnamese keyword to folder name
    folder_name = KEYWORD_MAPPING.get(keyword_lower)
    
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


def get_available_keywords() -> dict:
    """
    Get all available keywords organized by category.
    
    Returns:
        Dictionary mapping categories to lists of available keywords
    """
    return {
        "locations": ["dinh độc lập", "hồ gươm"],
        "styles": ["đông hồ", "sơn mài"],
        "objects": ["nón lá"]
    }