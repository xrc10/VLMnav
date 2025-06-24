import base64
import io
import os
import json
import datetime
import math
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import torch
import numpy as np
import cv2
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

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

def find_boundary_points(mask):
    """Find boundary points of the navigability mask."""
    # Use erosion to find boundaries
    kernel = np.ones((5, 5), np.uint8)
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
    boundary = mask.astype(np.uint8) - eroded
    
    # Get coordinates of boundary points
    boundary_points = np.argwhere(boundary > 0)
    
    # Filter points based on position (e.g., exclude bottom portion)
    height = mask.shape[0]
    boundary_points = boundary_points[boundary_points[:, 0] < height * 0.95]
    
    return boundary_points

def calculate_turning_degree(point, start_point):
    """Calculate the turning degree needed for the robot."""
    dx = point[1] - start_point[0]  # Convert from (row, col) to (x, y)
    dy = point[0] - start_point[1]  # point[0] is row (y), point[1] is column (x)
    
    angle = math.degrees(math.atan2(dx, -dy))  # Negative dy because y increases downward
    
    # Normalize angle to [-180, 180)
    if angle > 180:
        angle -= 360
    elif angle <= -180:
        angle += 360
        
    return angle

def filter_points_by_angle(points, start_point, min_angle=15):
    """Filter points based on a minimum angle between them."""
    if len(points) == 0:
        return []
    
    # Convert points to (row, col) format and calculate angles
    points_with_angles = [(point, calculate_turning_degree(point, start_point)) for point in points]
    
    # Sort points by angle
    points_with_angles.sort(key=lambda x: x[1])
    
    # Filter points with minimum angle difference
    filtered_points = [points_with_angles[0][0]]
    filtered_angles = [points_with_angles[0][1]]
    
    for point, angle in points_with_angles[1:]:
        # Check angle difference against all previously filtered points
        valid_point = True
        for prev_angle in filtered_angles:
            angle_diff = min((angle - prev_angle) % 360, (prev_angle - angle) % 360)
            if angle_diff < min_angle:
                valid_point = False
                break
        
        if valid_point:
            filtered_points.append(point)
            filtered_angles.append(angle)
    
    return filtered_points

def draw_action_proposals(image, boundary_points, start_point, number_size=15, navigability_mask=None, min_path_length=50, draw_degree=True, min_arrow_width=10, predicted_semantic_map=None, id2label=None):
    """Draw action proposals on the image and return the corresponding action info."""
    # Create a copy of the original image
    output_image = image.copy()
    height, width = image.shape[:2]
    
    # Keep track of valid points and their details for final numbering and info generation
    valid_points_details = []
    
    # Calculate center angle (0 degrees is straight ahead)
    center_x = width // 2
    
    # Iterate through the initially filtered boundary points
    for i, point in enumerate(boundary_points):
        end_point = (point[1], point[0])  # Convert from (row, col) to (x, y)
        
        # Calculate turning degree
        turning_degree = calculate_turning_degree(point, start_point)
        
        # Calculate angle from center (in degrees)
        angle_from_center = abs(turning_degree)
        
        # Adjust minimum path length based on angle
        angle_factor = 1.0 - (angle_from_center / 45.0) * 0.3
        angle_factor = max(0.7, min(1.0, angle_factor))  # Clamp between 0.7 and 1.0
        adjusted_min_path_length = int(min_path_length * angle_factor)
        
        # Find where the ray from start_point to end_point intersects the bottom of the image
        if start_point[1] != end_point[1]:  # Avoid division by zero
            t = (height - 1 - start_point[1]) / (end_point[1] - start_point[1])
            entry_x = int(start_point[0] + t * (end_point[0] - start_point[0]))
            entry_point = (entry_x, height - 1)
        else:
            entry_point = (start_point[0], height - 1)

        # Clamp entry point x-coordinate to be within image bounds
        entry_point = (max(0, min(width - 1, entry_point[0])), entry_point[1])
        
        # Store original end point for potentially drawing the arrow
        original_end_point = end_point
        
        # Track obstacle information
        obstacle_info = {
            'class_ids': set(),
            'class_names': set(),
            'blocking_point': None
        }
        
        # If navigability mask is provided, find the nearest navigable point along the path
        if navigability_mask is not None:
            # Get the direction from entry_point to end_point
            direction_x = end_point[0] - entry_point[0]
            direction_y = end_point[1] - entry_point[1]
            direction_length = math.sqrt(direction_x**2 + direction_y**2)
            
            if direction_length > 0:
                # Normalize the direction vector
                direction_x /= direction_length
                direction_y /= direction_length
                
                # Start from entry point and move along the direction
                current_x, current_y = float(entry_point[0]), float(entry_point[1])
                step_size = 1.0
                
                path_valid = True
                final_navigable_point = entry_point
                path_length = 0

                max_steps = int(direction_length / step_size) + 2

                for _ in range(max_steps):
                    dist_sq_to_end = (current_x - end_point[0])**2 + (current_y - end_point[1])**2
                    if dist_sq_to_end < step_size**2:
                        final_navigable_point = end_point
                        path_length = direction_length
                        break

                    next_x = current_x + direction_x * step_size
                    next_y = current_y + direction_y * step_size
                    path_length += step_size

                    pixel_x = int(round(next_x))
                    pixel_y = int(round(next_y))

                    if (pixel_y < 0 or pixel_y >= height or
                        pixel_x < 0 or pixel_x >= width):
                        final_navigable_point = (int(round(current_x)), int(round(current_y)))
                        break

                    is_valid = True
                    for dx in range(-min_arrow_width, min_arrow_width + 1):
                        for dy in range(-min_arrow_width, min_arrow_width + 1):
                            if dx*dx + dy*dy > min_arrow_width*min_arrow_width:
                                continue
                                
                            check_x = pixel_x + dx
                            check_y = pixel_y + dy
                            
                            if (check_x < 0 or check_x >= width or
                                check_y < 0 or check_y >= height):
                                continue
                                
                            if not navigability_mask[check_y, check_x]:
                                is_valid = False
                                # Record obstacle information if semantic map is available
                                if predicted_semantic_map is not None and id2label is not None:
                                    obstacle_class = predicted_semantic_map[check_y, check_x]
                                    obstacle_info['class_ids'].add(int(obstacle_class))
                                    obstacle_info['class_names'].add(id2label[obstacle_class])
                                    if obstacle_info['blocking_point'] is None:
                                        obstacle_info['blocking_point'] = (check_x, check_y)
                                break
                        if not is_valid:
                            break

                    if not is_valid:
                        final_navigable_point = (int(round(current_x)), int(round(current_y)))
                        break

                    current_x, current_y = next_x, next_y
                else:
                    final_navigable_point = (int(round(current_x)), int(round(current_y)))

                end_point = final_navigable_point
        
        # Recalculate path length based on the potentially adjusted end_point
        path_length = math.sqrt((end_point[0] - entry_point[0])**2 + (end_point[1] - entry_point[1])**2)
        
        # Store details for numbering and final action list
        valid_points_details.append({
            'mid_x': (entry_point[0] + end_point[0]) // 2,
            'mid_y': (entry_point[1] + end_point[1]) // 2,
            'end_point': end_point,
            'entry_point': entry_point,
            'turning_degree': turning_degree,
            'path_length': path_length,
            'is_blocked': path_length < adjusted_min_path_length,
            'obstacle_info': obstacle_info
        })

    # Post-process to identify paths that are shorter than their neighbors
    for i, details in enumerate(valid_points_details):
        # Get neighboring path lengths
        prev_length = valid_points_details[i-1]['path_length'] if i > 0 else details['path_length']
        next_length = valid_points_details[i+1]['path_length'] if i < len(valid_points_details)-1 else details['path_length']
        
        # Mark paths that are significantly shorter than their neighbors (at least 30% shorter)
        avg_neighbor_length = (prev_length + next_length) / 2
        details['shorter_than_neighbors'] = (not details['is_blocked'] and 
                                          details['path_length'] < 0.7 * avg_neighbor_length)

    # Draw arrows and numbers
    final_actions = []
    for i, details in enumerate(valid_points_details):
        action_number = i + 1
        mid_x = details['mid_x']
        mid_y = details['mid_y']
        end_point = details['end_point']
        entry_point = details['entry_point']
        turning_degree = details['turning_degree']
        path_length = details['path_length']
        is_blocked = details['is_blocked']
        shorter_than_neighbors = details.get('shorter_than_neighbors', False)
        obstacle_info = details['obstacle_info']
        
        # Determine arrow color based on status
        if is_blocked:
            arrow_color = (0, 0, 255)  # Red for blocked
            arrow_thickness = 6
        elif shorter_than_neighbors:
            arrow_color = (0, 255, 255)  # Yellow for warning
            arrow_thickness = 5
        else:
            arrow_color = (0, 255, 0)  # Green for clear
            arrow_thickness = 4
        
        # Draw arrow
        cv2.arrowedLine(
            output_image, 
            entry_point,
            end_point,
            arrow_color,
            arrow_thickness,
            tipLength=0.03
        )
        
        # Draw number circle
        circle_color = arrow_color
        cv2.circle(output_image, (mid_x, mid_y), number_size, (255, 255, 255), -1)
        cv2.circle(output_image, (mid_x, mid_y), number_size, circle_color, 2)
        
        text = str(action_number)
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        text_x = mid_x - text_size[0] // 2
        text_y = mid_y + text_size[1] // 2

        cv2.putText(
            output_image, 
            text, 
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (0, 0, 0),
            2
        )
        
        if draw_degree:
            degree_text = f"{turning_degree:.0f}°"
            degree_text_size, _ = cv2.getTextSize(degree_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            degree_text_x = mid_x - degree_text_size[0] // 2
            degree_text_y = mid_y + number_size + 15

            cv2.putText(
                output_image,
                degree_text,
                (degree_text_x, degree_text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                circle_color,
                1
            )

            # Add status text
            status_text = None
            if is_blocked:
                status_text = "BLOCKED"
            elif shorter_than_neighbors:
                status_text = "WARNING"
            
            if status_text:
                status_text_size, _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                status_text_x = mid_x - status_text_size[0] // 2
                status_text_y = mid_y + number_size + 35

                cv2.putText(
                    output_image,
                    status_text,
                    (status_text_x, status_text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    circle_color,
                    1
                )
                
                # Add obstacle information if available for blocked paths
                if is_blocked and obstacle_info['class_names']:
                    obstacle_text = f"Obstacles: {', '.join(obstacle_info['class_names'])}"
                    obstacle_text_size, _ = cv2.getTextSize(obstacle_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                    obstacle_text_x = mid_x - obstacle_text_size[0] // 2
                    obstacle_text_y = mid_y + number_size + 55

                    cv2.putText(
                        output_image,
                        obstacle_text,
                        (obstacle_text_x, obstacle_text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        circle_color,
                        1
                    )

        # Add to final action list
        final_actions.append({
            'action_number': action_number,
            'turning_degree': round(turning_degree, 1),
            'center_position': (int(mid_x), int(mid_y)),
            'boundary_point': (int(end_point[0]), int(end_point[1])),
            'path_length': int(path_length),
            'is_blocked': is_blocked,
            'shorter_than_neighbors': shorter_than_neighbors,
            'obstacle_info': {
                'class_ids': list(obstacle_info['class_ids']),
                'class_names': list(obstacle_info['class_names']),
                'blocking_point': obstacle_info['blocking_point']
            }
        })

    # Add summary at the top of the image
    summary_parts = []
    
    # Add blocked paths summary
    blocked_actions = [action for action in final_actions if action['is_blocked']]
    if blocked_actions:
        blocked_summary = []
        for action in blocked_actions:
            obstacle_names = action['obstacle_info']['class_names']
            if obstacle_names:
                blocked_summary.append(f"#{action['action_number']}: {', '.join(obstacle_names)}")
        if blocked_summary:
            summary_parts.append("Blocked paths - " + "; ".join(blocked_summary))
    
    # Add warning paths summary
    warning_actions = [action for action in final_actions if action['shorter_than_neighbors']]
    if warning_actions:
        warning_summary = [f"#{action['action_number']}" for action in warning_actions]
        if warning_summary:
            summary_parts.append("Warning: shorter paths - " + ", ".join(warning_summary))
    
    # Draw summary text if there are any warnings or blocked paths
    if summary_parts:
        summary_text = " | ".join(summary_parts)
        # Add semi-transparent background for better text visibility
        text_size, _ = cv2.getTextSize(summary_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        bg_height = text_size[1] + 20
        bg = output_image[0:bg_height, 0:width].copy()
        overlay = np.zeros_like(bg)
        cv2.rectangle(overlay, (0, 0), (width, bg_height), (255, 255, 255), -1)
        output_image[0:bg_height, 0:width] = cv2.addWeighted(bg, 0.2, overlay, 0.8, 0)
        
        cv2.putText(
            output_image,
            summary_text,
            (10, 30),  # Position the text near the top
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,  # Font scale
            (0, 0, 0),  # Black color
            2  # Thickness
        )
    
    return output_image, final_actions

def generate_action_proposals_from_image(image_data, min_angle=15, number_size=15, min_path_length=50, draw_degree=False, min_arrow_width=10):
    """Process an image and generate action proposals."""
    global device, processor, model
    
    # Get class ids for navigable regions
    id2label = model.config.id2label
    outdoor_labels = ["floor", "rug", "road", "sidewalk", "earth", "field", "sand", "dirt track", "land", "path", "runway"]
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
                                       predicted_semantic_map=predicted_semantic_map,
                                       id2label=id2label)
    
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
            min_arrow_width=min_arrow_width
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
            info_path = os.path.join(IMAGE_DIR, f"logs.json")
            with open(info_path, 'a') as f:
                f.write(json.dumps({
                    'action_info': action_info,
                    'timestamp': timestamp
                }) + '\n')

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

if __name__ == "__main__":
    # Configuration
    model_path = os.environ.get(
        'SEGMENTATION_MODEL_PATH', 
        "/data3/xu_ruochen/my_checkpoints/mask2former-swin-small-ade-semantic"
    )
    port = int(os.environ.get('PORT', 8077))
    
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
