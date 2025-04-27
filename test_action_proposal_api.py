import os
import base64
import requests
import glob
from PIL import Image
import io
import time
import json
import numpy as np
import cv2

"""
Action Proposal API

Endpoint: /generate_action_proposals

Description:
    This API generates navigation action proposals based on an input image. 
    It analyzes the scene and suggests possible navigation paths with turning degrees.

Input:
    JSON payload with the following fields:
    - image (str): Base64-encoded image data
    - min_angle (int): Minimum angle difference between adjacent proposals (default: 40)
    - number_size (int): Size of the number markers in the output visualization (default: 30)
    - min_path_length (int): Minimum path length to consider for proposals (default: 200)

Output:
    JSON response with the following fields:
    - actions (list): List of action proposals, each containing a dictionary with the following keys :
        - action_number (int): Index of the action proposal
        - turning_degree (float): Suggested turning angle in degrees
    - image (str): Base64-encoded image with visualized action proposals
"""

def test_action_proposal_api(
    api_url="http://10.8.25.28:8075/generate_action_proposals",
    image_pattern=".MVIMG_*",
    output_dir="./output/",
    min_angle=40,
    number_size=30,
    min_path_length=200,
    min_arrow_width=15,
    use_turn_left_right=False,
    use_turn_around=False
):
    """
    Test the action proposal API by sending local images and saving the results.
    
    Args:
        api_url: URL of the API endpoint
        image_pattern: Glob pattern to find input images
        output_dir: Directory to save output images and action JSONs
        min_angle: Minimum angle between proposals
        number_size: Size of the number markers
        min_path_length: Minimum path length for proposals
        min_arrow_width: Minimum arrow width for proposals
        use_turn_left_right: Whether to use turn left right for proposals
        use_turn_around: Whether to use turn around for proposals
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all matching image files
    image_files = glob.glob(image_pattern)
    print(f"Found {len(image_files)} images matching pattern '{image_pattern}'")
    
    if not image_files:
        print("No images found. Please check the pattern.")
        return
    
    for image_path in image_files:
        # Get filename without path
        filename = os.path.basename(image_path)
        output_image_path = os.path.join(output_dir, filename)
        
        # Construct JSON filename
        base_filename, _ = os.path.splitext(filename)
        json_filename = f"{base_filename}.json"
        output_json_path = os.path.join(output_dir, json_filename)
        
        print(f"Processing {filename}...")
        
        try:
            # Read image and convert to base64
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
            
            # Prepare request payload
            payload = {
                "image": image_base64,
                "min_angle": min_angle,
                "number_size": number_size,
                "min_path_length": min_path_length,
                "min_arrow_width": min_arrow_width,
                "use_turn_left_right": use_turn_left_right,
                "use_turn_around": use_turn_around
            }
            
            # Send request to API
            response = requests.post(api_url, json=payload, timeout=60)
            
            print(response.json().keys()) # dict_keys(['actions', 'image'])
            # actions [{'action_number': 0, 'turning_degree': 180.0}, {'action_number': 1, 'turning_degree': -56.9}, {'action_number': 2, 'turning_degree': -16.8}, {'action_number': 3, 'turning_degree': 23.3}, {'action_number': 4, 'turning_degree': 69.8}]
            # image is base64 encoded image in string

            if response.status_code == 200:
                # Get response data
                data = response.json()
                output_base64 = data["image"]
                actions = data["actions"]
                navigability_mask = data["navigability_mask"]
                
                # Decode base64 image
                output_bytes = base64.b64decode(output_base64)
                
                # Save output image
                with open(output_image_path, "wb") as f:
                    f.write(output_bytes)

                # Save navigation mask as image
                navigability_mask = np.array(navigability_mask, dtype=np.uint8)
                navigability_mask = navigability_mask * 255  # Convert to 0-255 range
                navigability_mask_path = os.path.join(output_dir, f"{base_filename}_mask.jpg")
                cv2.imwrite(navigability_mask_path, navigability_mask)
                print(f"✓ Saved navigability mask to {navigability_mask_path}")
                
                print(f"✓ Saved output image to {output_image_path}")

                # Save actions to JSON file
                with open(output_json_path, "w") as json_f:
                    json.dump(actions, json_f, indent=4)
                print(f"✓ Saved actions to {output_json_path}")

                print(f"  Actions: {actions}")
            else:
                print(f"✗ Error: API returned status code {response.status_code}")
                print(f"  Response: {response.text}")
        
        except Exception as e:
            print(f"✗ Error processing {filename}: {str(e)}")
    
    print("Processing complete!")

if __name__ == "__main__":
    # You can customize these parameters if needed
    API_URL = "http://10.8.25.28:8075/generate_action_proposals"
    # IMAGE_PATTERN = "./MVIMG_*"  # Pattern to match input images
    # IMAGE_PATTERN = "/data3/xu_ruochen/vlm_od_logs/*.jpg"
    IMAGE_PATTERN = "./ut_dog_depth_camera_rgb*.jpg"
    # IMAGE_PATTERN = "./ZED3_KSC_047355_L_P009301_png.rf.3557e43e49c61b09fdf2c479938de37e.jpg"
    # IMAGE_PATTERN = "./ZED3_KSC_047510_L_P009418_png.rf.30e99c2376bc8709863c1556be68e61c.jpg"
    OUTPUT_DIR = "./output/"
    MIN_ANGLE = 20
    NUMBER_SIZE = 20
    MIN_PATH_LENGTH = 50
    MIN_ARROW_WIDTH = 15

    # clean output folder
    if os.path.exists(OUTPUT_DIR):
        for file in os.listdir(OUTPUT_DIR):
            os.remove(os.path.join(OUTPUT_DIR, file))
    
    #iterate all images in the current directory
    for image_path in glob.glob(IMAGE_PATTERN):
        print(f"Processing {image_path}...")
        start_time = time.time()
        test_action_proposal_api(
            api_url=API_URL,
            image_pattern=image_path,
            output_dir=OUTPUT_DIR,
            min_angle=MIN_ANGLE,
            number_size=NUMBER_SIZE,
            min_path_length=MIN_PATH_LENGTH,
            min_arrow_width=MIN_ARROW_WIDTH,
            use_turn_left_right=True
        ) 
        end_time = time.time()
        print(f"Time taken: {end_time - start_time} seconds")
        break