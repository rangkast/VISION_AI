
import cv2
import numpy as np

class ImageFilter:
    def __init__(self, name):
        self.name = name

    def apply_no_filter(self, image):
        # Do nothing, return the original image
        return image

    def apply_custom_sharpening_filter(self, image):
        # 강한 샤프닝 효과를 주는 커널
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        sharpened_image = cv2.filter2D(image, -1, kernel)
        return sharpened_image

    def apply_laplacian_sharpening(self, image):
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        sharpened = cv2.convertScaleAbs(image - laplacian)
        return sharpened

    def apply_unsharp_mask(self, image, kernel_size=(3, 3), sigma=1.0, amount=0.3):
        blurred = cv2.GaussianBlur(image, kernel_size, sigma)
        sharpened = cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)
        return sharpened

    def adjust_brightness(self, image, beta_value=100):
        # Adjust the brightness by adding the beta_value to all pixels
        brightened_image = cv2.convertScaleAbs(image, alpha=1, beta=beta_value)
        return brightened_image

    def get_name(self):
        return self.name

def init_filter_models():
    models = []

    no_filter = ImageFilter("No Filter")
    models.append(no_filter)
    print(f"{no_filter.get_name()} init done")

    custom_filter = ImageFilter("Custom Sharpening Filter")
    models.append(custom_filter)
    print(f"{custom_filter.get_name()} init done")

    laplacian_filter = ImageFilter("Laplacian Sharpening")
    models.append(laplacian_filter)
    print(f"{laplacian_filter.get_name()} init done")

    unsharp_mask_filter = ImageFilter("Unsharp Mask")
    models.append(unsharp_mask_filter)
    print(f"{unsharp_mask_filter.get_name()} init done")

    brightness_filter = ImageFilter("Brightness Adjustment")
    models.append(brightness_filter)
    print(f"{brightness_filter.get_name()} init done")

    return models

def add_image_filter(image):
    filter_model = get_curr_filter()

    if filter_model.name == "No Filter":
        filtered_image = filter_model.apply_no_filter(image)
    elif filter_model.name == "Custom Sharpening Filter":
        filtered_image = filter_model.apply_custom_sharpening_filter(image)
    elif filter_model.name == "Laplacian Sharpening":
        filtered_image = filter_model.apply_laplacian_sharpening(image)
    elif filter_model.name == "Unsharp Mask":
        filtered_image = filter_model.apply_unsharp_mask(image)
    elif filter_model.name == "Brightness Adjustment":
        filtered_image = filter_model.adjust_brightness(image, beta_value=50)

    return filtered_image



filter_index = None
def set_curr_filter(data):
    global filter_index
    filter_index = data

def get_curr_filter():
    global filter_index
    return filter_index