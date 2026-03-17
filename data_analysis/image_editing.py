import cv2
import numpy as np
from pathlib import Path

def upsample_img(img:np.ndarray, like:np.ndarray):
    """Upsample the image to match the original image size."""
    return cv2.resize(img, (like.shape[1], like.shape[0]), interpolation=cv2.INTER_NEAREST)


def get_images_of_slices(org_img_path: Path, mask_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get the images of the slices and the mask.
    Args:
        org_img_path (path): Path to the original image.
        mask_path (path): Path to the mask image.
        Returns:
        mask (np.ndarray): Mask image.
        org_img (np.ndarray): Original image.
    returns:
        rounded_img (np.ndarray): Rounded image.
        org_img (np.ndarray): Original image.
        org_mask (np.ndarray): Original mask image.
    """

    assert mask_path.exists(), print(mask_path)
    assert org_img_path.exists(), print(org_img_path)

    org_img = cv2.imread(org_img_path)
    #org_img = cv2.cvtColor(org_img,)
    org_mask = cv2.imread(mask_path)

    #Resize the mask and process the image for the third plot
    resized_mask = cv2.resize(org_mask, (64, 64))  #64x64 pixels = 1mm x 1mm
    rounded_img = np.sum(resized_mask, axis=2)
    rounded_img[rounded_img > 0] = 1
    rounded_img[rounded_img <= 0] = 0

    return rounded_img, org_img, org_mask


def change_brightness(img, value=30):
    """Does what the function name says, changes the brightness of the image."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v,value)
    v[v > 255] = 255
    v[v < 0] = 0
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img