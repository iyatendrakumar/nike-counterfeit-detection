import cv2

def extract_roi(img, domain):
    """
    Extract Region of Interest based on domain.
    domain: 'shoes' or 'medicine'
    """
    edges = cv2.Canny(img, 50, 150)
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    h, w = img.shape
    best_box = None
    max_area = 0

    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch
        aspect_ratio = cw / ch if ch != 0 else 0

        if domain == "shoes":
            # Logo-like regions
            if 0.7 < aspect_ratio < 3.5 and area > max_area:
                best_box = (x, y, cw, ch)
                max_area = area

        elif domain == "medicine":
            # Text-strip regions
            if aspect_ratio > 2.5 and area > max_area:
                best_box = (x, y, cw, ch)
                max_area = area

    if best_box:
        x, y, cw, ch = best_box
        return img[y:y+ch, x:x+cw]

    # Fallback: return full image
    return img
