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
from tqdm import tqdm
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
    api_url="http://10.8.25.28:8077/generate_action_proposals",
    image_pattern=".MVIMG_*",
    output_dir="./output_video/",
    min_angle=40,
    number_size=30,
    min_path_length=200,
    min_arrow_width=15,
    use_turn_left_right=False,
    normal_fps=2.0,  # Normal frame rate
    fast_fps=4.0,   # Fast frame rate when no blocked/short paths
    slow_fps=1.0,   # Slow frame rate when blocked/short paths present
    consecutive_frames_for_speedup=5  # Number of consecutive frames without blocked/short paths to speed up
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
        normal_fps: Normal frame rate
        fast_fps: Fast frame rate when no blocked/short paths
        slow_fps: Slow frame rate when blocked/short paths present
        consecutive_frames_for_speedup: Number of consecutive frames without blocked/short paths to speed up
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all matching image files
    image_files = sorted(glob.glob(image_pattern))
    print(f"Found {len(image_files)} images matching pattern '{image_pattern}'")
    
    if not image_files:
        print("No images found. Please check the pattern.")
        return
    
    # Initialize video writer
    first_image = cv2.imread(image_files[0])
    height, width = first_image.shape[:2]
    video_path = os.path.join(output_dir, "output_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Initialize video writer with normal frame rate
    video_writer = cv2.VideoWriter(video_path, fourcc, normal_fps, (width, height))
    
    # Track consecutive frames without blocked/short paths
    consecutive_normal_frames = 0
    current_fps = normal_fps
    
    for image_path in tqdm(image_files):
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
                "use_turn_left_right": use_turn_left_right
            }
            
            # Send request to API
            response = requests.post(api_url, json=payload, timeout=60)
            
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
                
                # Check if there are any blocked or significantly shorter paths
                has_blocked_or_short = any(action.get('is_blocked', False) or 
                                         action.get('is_significantly_shorter', False) 
                                         for action in actions)
                
                # Update consecutive frames counter and adjust frame rate
                if has_blocked_or_short:
                    consecutive_normal_frames = 0
                    if current_fps != slow_fps:
                        current_fps = slow_fps
                        # Recreate video writer with new frame rate
                        video_writer.release()
                        video_writer = cv2.VideoWriter(video_path, fourcc, current_fps, (width, height))
                else:
                    consecutive_normal_frames += 1
                    if consecutive_normal_frames >= consecutive_frames_for_speedup and current_fps != fast_fps:
                        current_fps = fast_fps
                        # Recreate video writer with new frame rate
                        video_writer.release()
                        video_writer = cv2.VideoWriter(video_path, fourcc, current_fps, (width, height))
                
                # Add frame to video
                frame = cv2.imread(output_image_path)
                video_writer.write(frame)
                
            else:
                print(f"✗ Error: API returned status code {response.status_code}")
                print(f"  Response: {response.text}")
        
        except Exception as e:
            print(f"✗ Error processing {filename}: {str(e)}")
    
    # Release video writer
    video_writer.release()
    print(f"✓ Video saved to {video_path}")
    print("Processing complete!")

if __name__ == "__main__":
    # You can customize these parameters if needed
    API_URL = "http://10.8.25.28:8077/generate_action_proposals"
    IMAGE_PATTERN = "/data23/xu_ruochen/preprocessdatawithmllm/data/sidewalk_yolo/train/*.jpg"
    OUTPUT_DIR = "./output_video/"
    MIN_ANGLE = 10
    NUMBER_SIZE = 20
    MIN_PATH_LENGTH = 80
    MIN_ARROW_WIDTH = 5
    NORMAL_FPS = 2.0
    FAST_FPS = 4.0
    SLOW_FPS = 1.0
    CONSECUTIVE_FRAMES_FOR_SPEEDUP = 5

    # clean output folder
    if os.path.exists(OUTPUT_DIR):
        for file in os.listdir(OUTPUT_DIR):
            os.remove(os.path.join(OUTPUT_DIR, file))
    
    # Process all images and create video
    start_time = time.time()
    test_action_proposal_api(
        api_url=API_URL,
        image_pattern=IMAGE_PATTERN,
        output_dir=OUTPUT_DIR,
        min_angle=MIN_ANGLE,
        number_size=NUMBER_SIZE,
        min_path_length=MIN_PATH_LENGTH,
        min_arrow_width=MIN_ARROW_WIDTH,
        use_turn_left_right=True,
        normal_fps=NORMAL_FPS,
        fast_fps=FAST_FPS,
        slow_fps=SLOW_FPS,
        consecutive_frames_for_speedup=CONSECUTIVE_FRAMES_FOR_SPEEDUP
    ) 
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time} seconds")