
import cv2
import numpy as np
import os

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



# 필터 적용 함수들 정의
def apply_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 예시 필터 함수 (흑백 변환)
def convertToGrayscale(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grayscale_image

def apply_darken(image, intensity=0.5):
    darkened_image = image * intensity
    darkened_image = np.clip(darkened_image, 0, 255).astype(np.uint8)
    return darkened_image

def apply_custom_filter(image, filters):
    for filter_func in filters:
        image = filter_func(image)
    return image

def apply_brightness(image, factor=1.0):
    brightened_image = cv2.convertScaleAbs(image, alpha=factor, beta=0)
    return brightened_image

def apply_noise(image, noise_level=0.1):
    noise = np.random.randn(*image.shape) * 255 * noise_level
    noisy_image = image + noise.astype(np.uint8)
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image

def apply_blur(image, ksize=5):
    blurred_image = cv2.GaussianBlur(image, (ksize, ksize), 0)
    return blurred_image

def equalizeHistogram(image):
    # 히스토그램 평활화 적용
    equalized_image = cv2.equalizeHist(image)
    # 히스토그램 평활화 결과를 BGR로 변환
    bgr_image = cv2.cvtColor(equalized_image, cv2.COLOR_GRAY2BGR)
    return bgr_image


def sobel_filter(image):
    # Sobel 필터 적용
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)  # X축 경계
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)  # Y축 경계
    # 값을 절대값으로 변환하고 정규화
    sobelx = cv2.convertScaleAbs(sobelx)
    sobely = cv2.convertScaleAbs(sobely)
    # B, G, R 채널로 합치기
    bgr_image = cv2.merge([image, sobelx, sobely])
    return bgr_image


def addFilters_on_channel(image):
    # 다양한 필터 적용
    sharp_image = cv2.filter2D(image, -1, np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]]))  # 샤프닝 필터
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)  # 가우시안 블러
    edges = cv2.Canny(image, 100, 200)  # 엣지 디텍션
    # B, G, R 채널로 합치기
    bgr_image = cv2.merge([sharp_image, blurred_image, edges])  
    return bgr_image



def apply_filters_to_images(image_dir, filters):
    """
    주어진 경로의 모든 이미지에 필터를 적용하고, 동일 경로에 저장하는 함수
    
    Args:
        image_dir (str): 이미지가 저장된 디렉토리 경로
        filters (list): 이미지에 적용할 필터 함수들의 리스트
    """
    # 이미지 디렉토리 내의 모든 파일을 가져옴
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]

    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        image = cv2.imread(image_path)

        if image is None:
            print(f"이미지를 로드할 수 없습니다: {image_path}")
            continue

        # 필터 적용
        filtered_image = apply_custom_filter(image, filters)

        # 필터가 적용된 이미지를 원래 경로에 저장
        cv2.imwrite(image_path, filtered_image)
        # print(f"필터가 적용된 이미지를 저장했습니다: {image_path}")