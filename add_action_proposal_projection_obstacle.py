import torch
import numpy as np
import cv2
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
import matplotlib.pyplot as plt
import math

def generate_action_proposals(image_path, output_path, segmentation_model_path, min_angle=15, number_size=15, min_path_length=50, draw_degree=True, min_arrow_width=10):
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize SegFormer model and processor
    processor = AutoImageProcessor.from_pretrained(segmentation_model_path)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(segmentation_model_path).to(device)
    
    # Get class ids for navigable regions
    id2label = model.config.id2label
    navigability_class_ids = [id for id, label in id2label.items() 
                            if 'floor' in label.lower() or 'rug' in label.lower()]
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    img_np = np.array(image)
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    # Generate segmentation mask
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Post-process the segmentation output
    predicted_semantic_map = processor.post_process_semantic_segmentation(
        outputs, target_sizes=[image.size[::-1]])[0].cpu().numpy()
    
    # Create navigability mask
    navigability_mask = np.isin(predicted_semantic_map, navigability_class_ids)
    
    # For debugging: save the navigability mask
    plt.figure()
    plt.imshow(navigability_mask)
    plt.savefig('debug_navigability_mask.png')
    plt.close()
    
    print(f"Number of navigable pixels: {np.sum(navigability_mask)}")
    print(f"Image shape: {image.size}")
    print(f"Mask shape: {navigability_mask.shape}")
    
    # Find boundary points of the navigability mask
    boundary_points = find_boundary_points(navigability_mask)
    
    # Calculate the virtual starting point (outside the image)
    height, width = img_np.shape[:2]
    start_point = (width // 2, height + int(height * 0.2))  # Point below the image
    
    # Filter points based on minimum angle between them
    filtered_points = filter_points_by_angle(boundary_points, start_point, min_angle=min_angle)
    
    # Create action visualization image and get valid action indices
    output_image, valid_action_indices = draw_action_proposals(img_np, filtered_points, start_point, number_size=number_size, 
                                       navigability_mask=navigability_mask, min_path_length=min_path_length, draw_degree=draw_degree, min_arrow_width=min_arrow_width)
    
    # Save the final image
    cv2.imwrite(output_path, output_image)
    
    # Return the indices of the action proposals that were actually drawn
    return valid_action_indices

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
    """Calculate the turning degree needed for the robot.
    Returns angle in degrees where:
    0 = straight ahead
    negative = turn left
    positive = turn right
    """
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

def draw_action_proposals(image, boundary_points, start_point, number_size=15, navigability_mask=None, min_path_length=50, draw_degree=True, min_arrow_width=10, use_turn_left_right=False, use_turn_around=True):
    """Draw action proposals on the image and return the corresponding action info."""
    # Create a copy of the original image
    output_image = image.copy()
    height, width = image.shape[:2]
    
    # Define the "turn around" option position first (needed for action_number 0)
    turn_point_radius = number_size
    # Position closer to the corner, ensuring space for text
    turn_point_center_x = turn_point_radius + 10 
    turn_point_center_y = turn_point_radius + 10
    turn_point = (turn_point_center_x, turn_point_center_y)
    
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
        # Straight ahead (0 degrees) gets full length, 45 degrees gets 70% length
        angle_factor = 1.0 - (angle_from_center / 45.0) * 0.3
        angle_factor = max(0.7, min(1.0, angle_factor))  # Clamp between 0.7 and 1.0
        adjusted_min_path_length = int(min_path_length * angle_factor)
        
        # Find where the ray from start_point to end_point intersects the bottom of the image
        if start_point[1] != end_point[1]:  # Avoid division by zero
            # Ensure end_point[1] - start_point[1] is not zero to avoid division error
            # This check is technically redundant due to the outer if, but safer explicit check
            denominator = end_point[1] - start_point[1]
            if denominator == 0: continue # Should not happen due to outer if
            
            t = (height - 1 - start_point[1]) / denominator
            entry_x = int(start_point[0] + t * (end_point[0] - start_point[0]))
            entry_point = (entry_x, height - 1)
        else:
            entry_point = (start_point[0], height - 1)

        # Clamp entry point x-coordinate to be within image bounds
        entry_point = (max(0, min(width - 1, entry_point[0])), entry_point[1])
        
        # Store original end point for potentially drawing the arrow
        original_end_point = end_point
        
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
                # until we hit a non-navigable pixel or reach the end point
                current_x, current_y = float(entry_point[0]), float(entry_point[1]) # Use float for accumulation
                step_size = 1.0  # Step size for traversing the line
                
                path_valid = True
                final_navigable_point = entry_point # Default to entry if no steps taken or path invalid
                path_length = 0  # Track actual path length

                # Limit the number of steps to prevent infinite loops in edge cases
                max_steps = int(direction_length / step_size) + 2

                for _ in range(max_steps):
                    # Check if we're effectively at or past the end point
                    dist_sq_to_end = (current_x - end_point[0])**2 + (current_y - end_point[1])**2
                    if dist_sq_to_end < step_size**2:
                        final_navigable_point = end_point # Reached original end point while navigable
                        path_length = direction_length
                        break

                    # Move one step in the direction
                    next_x = current_x + direction_x * step_size
                    next_y = current_y + direction_y * step_size
                    path_length += step_size

                    # Round to get pixel coordinates for checking navigability
                    pixel_x = int(round(next_x))
                    pixel_y = int(round(next_y))

                    # Check boundaries
                    if (pixel_y < 0 or pixel_y >= height or
                        pixel_x < 0 or pixel_x >= width):
                        final_navigable_point = (int(round(current_x)), int(round(current_y)))
                        break # Went out of bounds

                    # Check if we've hit a non-navigable pixel or are too close to one
                    # Create a circular region around the current point to check for non-navigable pixels
                    is_valid = True
                    for dx in range(-min_arrow_width, min_arrow_width + 1):
                        for dy in range(-min_arrow_width, min_arrow_width + 1):
                            # Skip points outside the circle
                            if dx*dx + dy*dy > min_arrow_width*min_arrow_width:
                                continue
                                
                            check_x = pixel_x + dx
                            check_y = pixel_y + dy
                            
                            # Skip points outside image bounds
                            if (check_x < 0 or check_x >= width or
                                check_y < 0 or check_y >= height):
                                continue
                                
                            if not navigability_mask[check_y, check_x]:
                                is_valid = False
                                break
                        if not is_valid:
                            break

                    if not is_valid:
                        final_navigable_point = (int(round(current_x)), int(round(current_y))) # Use the last valid point
                        break # Hit non-navigable area or too close to one

                    # Update current position if still navigable and within bounds
                    current_x, current_y = next_x, next_y
                else:
                    # Loop finished without break, means we reached max_steps close to end_point
                    final_navigable_point = (int(round(current_x)), int(round(current_y)))

                end_point = final_navigable_point # Update end_point to the last valid navigable point found
        
        # Recalculate path length based on the potentially adjusted end_point
        path_length = math.sqrt((end_point[0] - entry_point[0])**2 + (end_point[1] - entry_point[1])**2)
        
        # Determine if the path is blocked (shorter than adjusted minimum length)
        is_blocked = path_length < adjusted_min_path_length
        
        # Skip this path if it's too short and raise a warning
        if is_blocked:
            print(f"Warning: Action proposal at angle {turning_degree:.1f}° is blocked (length: {path_length:.1f} < {adjusted_min_path_length:.1f})")
            continue
            
        # Draw arrow with appropriate color (green for navigable, red for blocked)
        arrow_color = (0, 255, 0) if not is_blocked else (0, 0, 255)  # Green for navigable, red for blocked
        
        cv2.arrowedLine(
            output_image, 
            entry_point,
            end_point,
            arrow_color,  # Use color based on navigability
            2,  # Line thickness
            tipLength=0.03
        )
        
        # Calculate position for the number (midpoint of visible arrow)
        mid_x = (entry_point[0] + end_point[0]) // 2
        mid_y = (entry_point[1] + end_point[1]) // 2
        
        # Store details for numbering and final action list
        valid_points_details.append({
            'mid_x': mid_x,
            'mid_y': mid_y,
            'end_point': end_point,  # Store end point for final action info
            'turning_degree': turning_degree,
            'path_length': path_length,
            'is_blocked': is_blocked
        })
    
    # Initialize the final action list
    final_actions = []
    
    if use_turn_around:
        # Draw the "turn around" option (action 0) representation at top left corner
        cv2.circle(output_image, turn_point, turn_point_radius, (255, 255, 255), -1)  # White background circle

        # Center the "0" text within the circle
        text_0 = "0"
        text_0_size, _ = cv2.getTextSize(text_0, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        text_0_x = turn_point_center_x - text_0_size[0] // 2
        text_0_y = turn_point_center_y + text_0_size[1] // 2
        cv2.putText(
            output_image,
            text_0,
            (text_0_x, text_0_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0), # Black text
            2
        )

        # Add descriptive text next to the "0" circle
        if draw_degree:
            turn_around_text = "turn around (180deg)"
        else:
            turn_around_text = "turn around"

        text_ta_x = turn_point_center_x + turn_point_radius + 5 # Position text to the right of the circle
        text_ta_y = turn_point_center_y + text_0_size[1] // 2

        cv2.putText(
            output_image,
            turn_around_text,
            (text_ta_x, text_ta_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, # Font scale
            (255, 0, 0), # Red text color
            2 # Thickness
        )
        
        # Add "turn around" action first (always action 0)
        final_actions.append({
            'action_number': 0,
            'turning_degree': 180.0,
            'center_position': (turn_point_center_x, turn_point_center_y),
            'boundary_point': None,  # No boundary point for turning around
            'path_length': 0,
            'is_blocked': False
        })
    elif use_turn_left_right:
        # Draw "L" and "R" indicators at top corners
        # Left corner "L"
        l_text = "L"
        l_text_size, _ = cv2.getTextSize(l_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        l_x = turn_point_radius + 10
        l_y = turn_point_radius + 10
        
        # Draw white circle for "L" with red boundary
        cv2.circle(output_image, (l_x, l_y), number_size, (255, 255, 255), -1)  # White background circle
        cv2.circle(output_image, (l_x, l_y), number_size, (255, 0, 0), 2)  # Red boundary
        
        # Center the "L" text within the circle
        l_text_x = l_x - l_text_size[0] // 2
        l_text_y = l_y + l_text_size[1] // 2
        cv2.putText(
            output_image,
            l_text,
            (l_text_x, l_text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0), # Black text
            2
        )
        
        # Right corner "R"
        r_text = "R"
        r_text_size, _ = cv2.getTextSize(r_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        r_x = width - turn_point_radius - 10 - r_text_size[0]
        r_y = turn_point_radius + 10
        
        # Draw white circle for "R" with red boundary
        cv2.circle(output_image, (r_x, r_y), number_size, (255, 255, 255), -1)  # White background circle
        cv2.circle(output_image, (r_x, r_y), number_size, (255, 0, 0), 2)  # Red boundary
        
        # Center the "R" text within the circle
        r_text_x = r_x - r_text_size[0] // 2
        r_text_y = r_y + r_text_size[1] // 2
        cv2.putText(
            output_image,
            r_text,
            (r_text_x, r_text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0), # Black text
            2
        )
        
        # Add turn left and right actions
        final_actions.append({
            'action_number': "L",
            'turning_degree': -90.0,  # Left turn
            'center_position': (l_x, l_y),
            'boundary_point': None
        })
        final_actions.append({
            'action_number': "R",
            'turning_degree': 90.0,  # Right turn
            'center_position': (r_x, r_y),
            'boundary_point': None
        })
    
    # Draw numbers and turning degrees for valid paths and build final action list
    for i, details in enumerate(valid_points_details):
        action_number = i + 1  # Start numbering from 1 for actual actions
        mid_x = details['mid_x']
        mid_y = details['mid_y']
        end_point = details['end_point']
        turning_degree = details['turning_degree']
        path_length = details['path_length']
        is_blocked = details['is_blocked']
        
        # Draw number in circle with white background
        cv2.circle(output_image, (mid_x, mid_y), number_size, (255, 255, 255), -1)  # White background
        
        # Adjust text position based on number of digits for better centering
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
            (0, 0, 0),  # Black text
            2
        )
        
        # Add turning degree text if requested
        if draw_degree:
             degree_text = f"{turning_degree:.0f}°"
             # Position degree text below the number circle
             degree_text_size, _ = cv2.getTextSize(degree_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
             degree_text_x = mid_x - degree_text_size[0] // 2 # Center below circle
             degree_text_y = mid_y + number_size + 15 # Below circle + padding

             cv2.putText(
                 output_image,
                 degree_text,
                 (degree_text_x, degree_text_y),
                 cv2.FONT_HERSHEY_SIMPLEX,
                 0.5,
                 (255, 0, 0) if is_blocked else (0, 255, 0),  # Red for blocked, green for navigable
                 1 # Thinner line for degree text
             )

        # convert all numbers to int
        mid_x = int(mid_x)
        mid_y = int(mid_y)
        end_point = (int(end_point[0]), int(end_point[1]))

        # Add to final action list with the additional information
        final_actions.append({
            'action_number': action_number,
            'turning_degree': round(turning_degree, 1),
            'center_position': (mid_x, mid_y),  # Add center position
            'boundary_point': end_point,  # Add boundary point
            'path_length': path_length,
            'is_blocked': is_blocked
        })
    
    # Return the image and the final list of actions
    return output_image, final_actions

if __name__ == "__main__":
    input_image = "./MVIMG_1.jpg"
    output_image = "./indoor_actions.jpg"
    segmentation_model_path = "/data3/xu_ruochen/my_checkpoints/mask2former-swin-small-ade-semantic"
    min_angle = 40
    number_size = 30
    min_path_length = 50  # Minimum path length in pixels
    draw_degree = False
    min_arrow_width = 10
    actions = generate_action_proposals(input_image, output_image, segmentation_model_path, 
                                     min_angle, number_size, min_path_length, draw_degree, min_arrow_width)
    print(actions)
