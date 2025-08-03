from PIL import Image
from think_n_blend.schemas import BoundingBox, RelativePosition

def compute_target_bounding_box(image_path: str, reference_box: BoundingBox, relative_position: RelativePosition) -> BoundingBox:
    """
    Computes the target bounding box for the new object based on the reference box
    and the relative position.
    """
    img_width, img_height = Image.open(image_path).size
    x1, y1, x2, y2 = reference_box
    ref_width = x2 - x1
    ref_height = y2 - y1
    
    x1_new, y1_new, x2_new, y2_new = 0, 0, 0, 0

    if relative_position == "top":
        y1_new = max(0, y1 - ref_height)
        x1_new, x2_new = x1, x2
        y2_new = y1
    elif relative_position == "bottom":
        y1_new = y2
        x1_new, x2_new = x1, x2
        y2_new = min(img_height, y2 + ref_height)
    elif relative_position == "left":
        x1_new = max(0, x1 - ref_width)
        y1_new, y2_new = y1, y2
        x2_new = x1
    elif relative_position == "right":
        x1_new = x2
        y1_new, y2_new = y1, y2
        x2_new = min(img_width, x2 + ref_width)
    
    return (x1_new, y1_new, x2_new, y2_new)
