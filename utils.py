import base64
import numpy as np
from PIL import Image
from io import BytesIO


def base64_to_image(base64_string):
    # Decode the Base64 string to bytes
    image_bytes = base64.b64decode(base64_string)
    
    # Create an Image object from the bytes
    image = Image.open(BytesIO(image_bytes))

    return image