# Image Selection Constants
IMAGE_EXTENSIONS = {
    ".png": 3, 
    ".jpg": 1, 
    ".jpeg": 1, 
    ".svg": 4, 
    ".gif": 2, 
    ".webp": 2
}

IMAGE_PRIORITY_KEYWORDS = {
    "architecture", "system", "diagram", "flow", "design", "overview"
}

IMAGE_IGNORE_KEYWORDS = {
    "icon", "badge", "logo", "button", "screenshot", "avatar"
}

# Link Transformation Constants
ABSOLUTE_URL_PREFIXES = ("http://", "https://", "#", "mailto:", "data:")
VALID_URL_SCHEMES = ("http", "https")
