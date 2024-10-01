import os
from matplotlib import pyplot as plt
import torch
import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import cv2
import uuid
import logging

logger = logging.getLogger('uvicorn.error')
logger.setLevel(logging.DEBUG)

def apply_mask(mask, image) -> str: # later move to util
    # Define the color with an alpha value
    color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    
    # Create a new image with the same size as the original
    original_image = image  # Load the original image
    if original_image is None:
        raise ValueError(f"Could not open or find the image: {image}")
        
    # Ensure the original image has an alpha channel
    if original_image.shape[2] != 4:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2BGRA)

    # Create a mask image
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    
    # Find contours if needed
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
    mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
    
    # Convert mask_image to uint8 format for saving
    mask_image = (mask_image * 255).astype(np.uint8)

    # Create the final output image
    output_image = np.zeros_like(original_image)  # Create an empty image with the same size and channels as the original
    output_image[..., 3] = mask_image[..., 3]  # Set the alpha channel to the mask's alpha
    output_image[..., :3] = original_image[..., :3] * (mask_image[..., :3] > 0)  # Use the original colors where the mask is applied

    # Create file name
    new_file_name = f"{uuid.uuid4()}.png"
    new_file_path = os.path.join(os.path.dirname("./result/"), new_file_name)
    # Save the final masked image
    cv2.imwrite(new_file_path, output_image)
    print("Masked image saved at: /result.png") 
    
    return new_file_path


def segSam2(path: str) -> str:
    image =  cv2.imread(path,cv2.IMREAD_UNCHANGED)
    print(f"Image size: {image.size}")
    logger.debug(f"Image size: {image.size}")
    sam2_checkpoint = "./weights/sam2.1_hiera_tiny.pt"

    model_cfg = "/configs/sam2.1/sam2.1_hiera_t.yaml"
    
    # if torch.cuda.is_available():
    #     device = torch.device("cuda")
    # elif torch.backends.mps.is_available():
    #     device = torch.device("mps")
    # else:
    device = torch.device("cpu")
    print(f"using device: {device}")
    
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)

    predictor = SAM2ImagePredictor(sam2_model)
    predictor.set_image(image)
    
    input_point = np.array([[500, 375]])
    input_label = np.array([1])
    
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    sorted_ind = np.argsort(scores)[::-1]
    masks = masks[sorted_ind]
    scores = scores[sorted_ind]
    logits = logits[sorted_ind]
    
    result = apply_mask(masks[0],image)
    
    return result

def main():
    print("This is the main function.")
    img =  cv2.imread("./uploads/show.jpg",cv2.IMREAD_UNCHANGED)
    segSam2(img)
    # Add your main logic here

if __name__ == "__main__":
    main()