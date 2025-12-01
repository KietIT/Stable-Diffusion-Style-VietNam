import os
from PIL import Image
from typing import Optional

def insert_object(background_path: str, object_path: str, output_dir: str = "output") -> Optional[str]:
    """
    Insert an object image onto a background image.
    The object is resized to 30% of background size and placed at bottom-center.
    
    Args:
        background_path: Path to background (stylized) image
        object_path: Path to object image to insert
        output_dir: Directory to save output (default: "output")
    
    Returns:
        Path to the final composite image, or None if failed
    """
    try:
        # Load images
        if not os.path.exists(background_path):
            print(f"Error: Background image not found: {background_path}")
            return None
        
        if not os.path.exists(object_path):
            print(f"Error: Object image not found: {object_path}")
            return None
        
        background = Image.open(background_path).convert("RGBA")
        obj_image = Image.open(object_path).convert("RGBA")
        
        # Calculate object size (30% of background)
        bg_width, bg_height = background.size
        obj_width = int(bg_width * 0.3)
        
        # Resize object maintaining aspect ratio
        obj_aspect_ratio = obj_image.size[1] / obj_image.size[0]
        obj_height = int(obj_width * obj_aspect_ratio)
        obj_resized = obj_image.resize((obj_width, obj_height), Image.Resampling.LANCZOS)
        
        # Calculate position (bottom-center)
        x_position = (bg_width - obj_width) // 2
        y_position = bg_height - obj_height - int(bg_height * 0.05)  # 5% margin from bottom
        
        # Create a copy of background to paste on
        result = background.copy()
        
        # Paste object with transparency
        result.paste(obj_resized, (x_position, y_position), obj_resized)
        
        # Convert back to RGB for saving
        result_rgb = result.convert("RGB")
        
        # Generate output filename
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_path = os.path.join(root_dir, output_dir)
        os.makedirs(output_path, exist_ok=True)
        
        bg_basename = os.path.splitext(os.path.basename(background_path))[0]
        obj_basename = os.path.splitext(os.path.basename(object_path))[0]
        output_filename = f"{bg_basename}_with_{obj_basename}.png"
        output_filepath = os.path.join(output_path, output_filename)
        
        # Save the result
        result_rgb.save(output_filepath, "PNG")
        print(f"Object inserted successfully! Saved to: {output_filepath}")
        
        return output_filepath
        
    except Exception as e:
        print(f"Error inserting object: {e}")
        return None


def insert_object_at_position(
    background_path: str,
    object_path: str,
    position: tuple = None,
    scale: float = 0.3,
    output_dir: str = "output"
) -> Optional[str]:
    """
    Advanced version: Insert object at specific position with custom scale.
    
    Args:
        background_path: Path to background image
        object_path: Path to object image
        position: (x, y) tuple for object position. If None, use bottom-center
        scale: Scale factor for object size (default: 0.3 = 30%)
        output_dir: Directory to save output
    
    Returns:
        Path to the final composite image, or None if failed
    """
    try:
        background = Image.open(background_path).convert("RGBA")
        obj_image = Image.open(object_path).convert("RGBA")
        
        bg_width, bg_height = background.size
        obj_width = int(bg_width * scale)
        obj_aspect_ratio = obj_image.size[1] / obj_image.size[0]
        obj_height = int(obj_width * obj_aspect_ratio)
        obj_resized = obj_image.resize((obj_width, obj_height), Image.Resampling.LANCZOS)
        
        # Determine position
        if position is None:
            x_position = (bg_width - obj_width) // 2
            y_position = bg_height - obj_height - int(bg_height * 0.05)
        else:
            x_position, y_position = position
        
        result = background.copy()
        result.paste(obj_resized, (x_position, y_position), obj_resized)
        result_rgb = result.convert("RGB")
        
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_path = os.path.join(root_dir, output_dir)
        os.makedirs(output_path, exist_ok=True)
        
        bg_basename = os.path.splitext(os.path.basename(background_path))[0]
        obj_basename = os.path.splitext(os.path.basename(object_path))[0]
        output_filename = f"{bg_basename}_with_{obj_basename}_custom.png"
        output_filepath = os.path.join(output_path, output_filename)
        
        result_rgb.save(output_filepath, "PNG")
        print(f"Object inserted at custom position! Saved to: {output_filepath}")
        
        return output_filepath
        
    except Exception as e:
        print(f"Error inserting object: {e}")
        return None