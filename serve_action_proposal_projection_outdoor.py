import base64
import io
import os
import json
import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import torch
import numpy as np
import cv2
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
import matplotlib.pyplot as plt
from add_action_proposal_projection_fixed import (
    find_boundary_points,
    filter_points_by_angle,
    draw_action_proposals,
    calculate_turning_degree
)

app = Flask(__name__)
# Enable CORS for all routes
CORS(app)

# Initialize rate limiter
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# Global variables to store the loaded model
device = None
processor = None
model = None

# Directory to save logs and images
LOG_DIR = "/data3/xu_ruochen/vlmnav_vlm_result_log"

# Directory to save images
IMAGE_DIR = "/data3/xu_ruochen/vlmnav_action_proposal_log"

def ensure_directory_exists(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_model(model_path):
    """Load the segmentation model once at startup."""
    global device, processor, model
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(model_path)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_path).to(device)
    
    return model.config.id2label

def generate_action_proposals_from_image(image_data, min_angle=15, number_size=15, min_path_length=50, draw_degree=False, min_arrow_width=10, use_turn_left_right=False):
    """Process an image and generate action proposals."""
    global device, processor, model
    
    # Get class ids for navigable regions
    id2label = model.config.id2label
    # print(f"id2label: {id2label}")
    # {0: 'wall', 1: 'building', 2: 'sky', 3: 'floor', 4: 'tree', 5: 'ceiling', 6: 'road', 7: 'bed ', 8: 'windowpane', 9: 'grass', 10: 'cabinet', 11: 'sidewalk', 12: 'person', 13: 'earth', 14: 'door', 15: 'table', 16: 'mountain', 17: 'plant', 18: 'curtain', 19: 'chair', 20: 'car', 21: 'water', 22: 'painting', 23: 'sofa', 24: 'shelf', 25: 'house', 26: 'sea', 27: 'mirror', 28: 'rug', 29: 'field', 30: 'armchair', 31: 'seat', 32: 'fence', 33: 'desk', 34: 'rock', 35: 'wardrobe', 36: 'lamp', 37: 'bathtub', 38: 'railing', 39: 'cushion', 40: 'base', 41: 'box', 42: 'column', 43: 'signboard', 44: 'chest of drawers', 45: 'counter', 46: 'sand', 47: 'sink', 48: 'skyscraper', 49: 'fireplace', 50: 'refrigerator', 51: 'grandstand', 52: 'path', 53: 'stairs', 54: 'runway', 55: 'case', 56: 'pool table', 57: 'pillow', 58: 'screen door', 59: 'stairway', 60: 'river', 61: 'bridge', 62: 'bookcase', 63: 'blind', 64: 'coffee table', 65: 'toilet', 66: 'flower', 67: 'book', 68: 'hill', 69: 'bench', 70: 'countertop', 71: 'stove', 72: 'palm', 73: 'kitchen island', 74: 'computer', 75: 'swivel chair', 76: 'boat', 77: 'bar', 78: 'arcade machine', 79: 'hovel', 80: 'bus', 81: 'towel', 82: 'light', 83: 'truck', 84: 'tower', 85: 'chandelier', 86: 'awning', 87: 'streetlight', 88: 'booth', 89: 'television receiver', 90: 'airplane', 91: 'dirt track', 92: 'apparel', 93: 'pole', 94: 'land', 95: 'bannister', 96: 'escalator', 97: 'ottoman', 98: 'bottle', 99: 'buffet', 100: 'poster', 101: 'stage', 102: 'van', 103: 'ship', 104: 'fountain', 105: 'conveyer belt', 106: 'canopy', 107: 'washer', 108: 'plaything', 109: 'swimming pool', 110: 'stool', 111: 'barrel', 112: 'basket', 113: 'waterfall', 114: 'tent', 115: 'bag', 116: 'minibike', 117: 'cradle', 118: 'oven', 119: 'ball', 120: 'food', 121: 'step', 122: 'tank', 123: 'trade name', 124: 'microwave', 125: 'pot', 126: 'animal', 127: 'bicycle', 128: 'lake', 129: 'dishwasher', 130: 'screen', 131: 'blanket', 132: 'sculpture', 133: 'hood', 134: 'sconce', 135: 'vase', 136: 'traffic light', 137: 'tray', 138: 'ashcan', 139: 'fan', 140: 'pier', 141: 'crt screen', 142: 'plate', 143: 'monitor', 144: 'bulletin board', 145: 'shower', 146: 'radiator', 147: 'glass', 148: 'clock', 149: 'flag'}
    outdoor_labels = ["floor", "rug", "road", "sidewalk", "earth", "field", "sand", "dirt track", "land"]
    navigability_class_ids = [id for id, label in id2label.items() 
                            if label in outdoor_labels]
    
    # Process the image
    inputs = processor(images=image_data, return_tensors="pt").to(device)
    
    # Generate segmentation mask
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Post-process the segmentation output
    predicted_semantic_map = processor.post_process_semantic_segmentation(
        outputs, target_sizes=[image_data.size[::-1]])[0].cpu().numpy()
    
    # Create navigability mask
    navigability_mask = np.isin(predicted_semantic_map, navigability_class_ids)
    
    # Convert PIL Image to numpy array
    img_np = np.array(image_data)
    
    # Find boundary points of the navigability mask
    boundary_points = find_boundary_points(navigability_mask)
    
    # Calculate the virtual starting point (outside the image)
    height, width = img_np.shape[:2]
    start_point = (width // 2, height + int(height * 0.2))  # Point below the image
    
    # Filter points based on minimum angle between them
    filtered_points = filter_points_by_angle(boundary_points, start_point, min_angle=min_angle)
    
    # Create action visualization image and get the final filtered/numbered action info
    output_image, action_info = draw_action_proposals(img_np, filtered_points, start_point, 
                                       number_size=number_size, 
                                       navigability_mask=navigability_mask,
                                       min_path_length=min_path_length,
                                       draw_degree=draw_degree,
                                       min_arrow_width=min_arrow_width,
                                       use_turn_left_right=use_turn_left_right)
    
    return output_image, action_info, navigability_mask

@app.route('/generate_action_proposals', methods=['POST'])
@limiter.limit("200 per minute")
def process_image():
    """API endpoint to process images and generate action proposals."""
    try:
        # Get request data
        data = request.json
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Get parameters
        min_angle = data.get('min_angle', 15)
        number_size = data.get('number_size', 15)
        min_path_length = data.get('min_path_length', 50)
        draw_degree = data.get('draw_degree', False)
        save_image = data.get('save_image', False)
        min_arrow_width = data.get('min_arrow_width', 10)
        use_turn_left_right = data.get('use_turn_left_right', False)
        
        # Decode base64 image
        try:
            image_bytes = base64.b64decode(data['image'])
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        except Exception as e:
            return jsonify({'error': f'Invalid image data: {str(e)}'}), 400
        
        # Process the image
        output_image, action_info, navigability_mask = generate_action_proposals_from_image(
            image, 
            min_angle=min_angle, 
            number_size=number_size,
            min_path_length=min_path_length,
            draw_degree=draw_degree,
            min_arrow_width=min_arrow_width,
            use_turn_left_right=use_turn_left_right
        )
        
        # if save_image is True, save the image
        if save_image:
            # Create the image directory if it doesn't exist
            os.makedirs(IMAGE_DIR, exist_ok=True)
            
            # Generate a timestamp for the filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            
            # Save the image
            image_path = os.path.join(IMAGE_DIR, f"{timestamp}.jpg")
            cv2.imwrite(image_path, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))

            # save the action info
            action_info_path = os.path.join(IMAGE_DIR, f"logs.json")
            with open(action_info_path, 'a') as f:
                f.write(json.dumps(action_info) + '\n')

        # Convert output image to base64
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
        output_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'image': output_base64,
            'navigability_mask': navigability_mask.tolist(),
            'navigability_mask_shape': navigability_mask.shape,
            'navigability_mask_dtype': str(navigability_mask.dtype),
            'actions': action_info
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/save_result', methods=['POST'])
@limiter.limit("100 per minute")
def save_result():
    """API endpoint to save VLM navigation results."""
    try:
        # Get request data
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Extract required fields
        if 'image' not in data:
            return jsonify({'error': 'Image data is required'}), 400
        if 'vlm_output' not in data:
            return jsonify({'error': 'VLM output is required'}), 400
        if 'action_number' not in data:
            return jsonify({'error': 'Action number is required'}), 400
            
        # Ensure the log directory exists
        ensure_directory_exists(LOG_DIR)
        
        # Generate a timestamp for the filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        # Save the image
        try:
            image_bytes = base64.b64decode(data['image'])
            image_path = os.path.join(LOG_DIR, f"{timestamp}.jpg")
            with open(image_path, 'wb') as f:
                f.write(image_bytes)
        except Exception as e:
            return jsonify({'error': f'Failed to save image: {str(e)}'}), 500
        
        # Prepare the log entry
        log_entry = {
            'timestamp': timestamp,
            'image_path': image_path,
            'vlm_output': data['vlm_output'],
            'action_number': data['action_number'],
        }
        
        # Add optional fields if present
        for field in ['episode_id', 'step_id', 'additional_info']:
            if field in data:
                log_entry[field] = data[field]
        
        # Append to the JSONL file
        log_file = os.path.join(LOG_DIR, 'vlmnav_results.jsonl')
        try:
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            return jsonify({'error': f'Failed to write to log file: {str(e)}'}), 500
        
        return jsonify({
            'success': True,
            'message': 'Result saved successfully',
            'image_path': image_path
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    # Configuration
    model_path = os.environ.get(
        'SEGMENTATION_MODEL_PATH', 
        "/data3/xu_ruochen/my_checkpoints/mask2former-swin-small-ade-semantic"
    )
    port = int(os.environ.get('PORT', 8075))
    
    # Load model at startup
    print(f"Loading segmentation model from {model_path}...")
    id2label = load_model(model_path)
    print(f"Model loaded successfully with {len(id2label)} classes.")
    
    # Ensure log directory exists
    ensure_directory_exists(LOG_DIR)
    print(f"Log directory: {LOG_DIR}")
    
    # Start server
    print(f"Starting server on port {port}...")
    app.run(host='0.0.0.0', port=port, threaded=True)
