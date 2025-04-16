import torch
import numpy as np
import cv2
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
import matplotlib.pyplot as plt
import math

def local_to_global(position, rotation, local_point):
    """Convert a point from local coordinates to global coordinates."""
    # Convert quaternion to rotation matrix
    w, x, y, z = rotation
    rotation_matrix = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])
    
    # Apply rotation and translation
    global_point = rotation_matrix @ np.array(local_point) + position
    return global_point

def agent_frame_to_image_coords(point, resolution, focal_length):
    """Project a point in agent frame to image coordinates."""
    x, y, z = point
    print(f"Projecting point: ({x}, {y}, {z})")  # Debug print
    if z >= 0:  # Point is behind the camera
        print(f"Point rejected: z >= 0")  # Debug print
        return None
    
    # Project to image plane
    px = int(-focal_length * x / z + resolution[1] / 2)
    py = int(-focal_length * y / z + resolution[0] / 2)
    print(f"Projected to pixel: ({px}, {py})")  # Debug print
    
    if 0 <= px < resolution[1] and 0 <= py < resolution[0]:
        return (px, py)
    print(f"Point rejected: outside image bounds")  # Debug print
    return None

def find_intersections(x1, y1, x2, y2, W, H):
    """Find intersections of a line segment with image boundaries."""
    # If both points are within the image, return them directly
    if (0 <= x1 < W and 0 <= y1 < H and 0 <= x2 < W and 0 <= y2 < H):
        return [(x1, y1), (x2, y2)]
    
    def line_intersection(p1, p2, p3, p4):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denominator == 0:
            return None
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator
        if 0 <= t <= 1:
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            return (x, y)
        return None

    # Image boundaries
    boundaries = [
        ((0, 0), (W-1, 0)),      # Top
        ((W-1, 0), (W-1, H-1)),  # Right
        ((W-1, H-1), (0, H-1)),  # Bottom
        ((0, H-1), (0, 0))       # Left
    ]
    
    intersections = []
    for boundary in boundaries:
        intersection = line_intersection((x1, y1), (x2, y2), boundary[0], boundary[1])
        if intersection:
            intersections.append(intersection)
    
    if len(intersections) >= 2:
        return intersections[:2]
    elif len(intersections) == 1:
        # If we have one intersection and one point inside the image, use both
        if 0 <= x1 < W and 0 <= y1 < H:
            return [(x1, y1), intersections[0]]
        elif 0 <= x2 < W and 0 <= y2 < H:
            return [intersections[0], (x2, y2)]
    return None

def check_navigability(mask, x, y, window_size=5):
    """Check if a point is navigable by looking at a window around it."""
    h, w = mask.shape
    x1, x2 = max(0, x-window_size//2), min(w, x+window_size//2+1)
    y1, y2 = max(0, y-window_size//2), min(h, y+window_size//2+1)
    window = mask[y1:y2, x1:x2]
    navigable_ratio = np.sum(window) / window.size
    print(f"Checking navigability at ({x}, {y}): ratio = {navigable_ratio:.2f}")  # Debug print
    return navigable_ratio > 0.3  # More lenient threshold

def check_navigable_path(mask, start_px, end_px, num_samples=20):
    """Check if path is navigable by sampling points along it."""
    x_coords = np.linspace(start_px[0], end_px[0], num_samples)
    y_coords = np.linspace(start_px[1], end_px[1], num_samples)
    
    navigable_count = 0
    for x, y in zip(x_coords, y_coords):
        x, y = int(x), int(y)
        if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:
            if check_navigability(mask, x, y):
                navigable_count += 1
    
    return navigable_count / num_samples > 0.7  # At least 70% of path should be navigable

def generate_action_proposals(image_path, output_path, segmentation_model_path):
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
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    # Generate segmentation mask
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Post-process the segmentation output
    predicted_semantic_map = processor.post_process_semantic_segmentation(
        outputs, target_sizes=[image.size[::-1]])[0].cpu().numpy()
    
    # Create navigability mask
    navigability_mask = np.isin(predicted_semantic_map, navigability_class_ids)
    
    # After creating navigability_mask
    plt.figure()
    plt.imshow(navigability_mask)
    plt.savefig('debug_navigability_mask.png')
    plt.close()
    
    print(f"Number of navigable pixels: {np.sum(navigability_mask)}")
    print(f"Image shape: {image.size}")
    print(f"Mask shape: {navigability_mask.shape}")
    
    if np.sum(navigability_mask) / (navigability_mask.shape[0] * navigability_mask.shape[1]) < 0.01:
        print("Warning: Very small navigable area detected")
    
    # Configuration for action proposals
    fov = 90  # degrees
    resolution = (1080, 1920)
    focal_length = resolution[1] / (2 * np.tan(np.deg2rad(fov/2)))
    print(f"Resolution: {resolution}, FOV: {fov}, Focal length: {focal_length}")
    max_action_dist = 2.0
    min_angle = fov/4  # For action spacing
    
    # Generate initial action proposals
    center_px = (resolution[1]//2, resolution[0]//2)
    print(f"Center pixel: {center_px}")
    actions = []
    
    # Generate actions at different angles
    print(f"Starting action generation with FOV: {fov}")  # Debug print
    for theta in np.linspace(-fov/2 * 0.9, fov/2 * 0.9, 7):
        theta_rad = np.deg2rad(theta)
        print(f"\nChecking angle: {np.rad2deg(theta_rad):.2f} degrees")  # Debug print
        
        # Project a point at this angle
        point = [2 * np.sin(theta_rad), 0, -2 * np.cos(theta_rad)]
        print(f"Created point in agent frame: {point}")  # Debug print
        
        end_px = agent_frame_to_image_coords(point, resolution, focal_length)
        print(f"End pixel coordinates: {end_px}")  # Debug print
        
        if end_px is None:
            print("Skipping: end point projection failed")  # Debug print
            continue
        
        # Find intersections
        intersections = find_intersections(center_px[0], center_px[1], end_px[0], end_px[1], resolution[1], resolution[0])
        print(f"Found intersections: {intersections}")  # Debug print
        
        if intersections is None:
            print("Skipping: no valid intersections found")  # Debug print
            continue
        
        (x1, y1), (x2, y2) = intersections
        num_points = max(abs(x2 - x1), abs(y2 - y1)) + 1
        print(f"Checking {num_points} points along the line")  # Debug print
        
        # Create the arrays of coordinates here
        x_coords = np.linspace(x1, x2, num_points)
        y_coords = np.linspace(y1, y2, num_points)
        
        # Check navigability along the line
        max_dist = 0
        for i in range(num_points):
            x = int(x_coords[i])
            y = int(y_coords[i])
            if not check_navigability(navigability_mask, x, y):
                print(f"Non-navigable point found at ({x}, {y})")  # Debug print
                break
            max_dist = min(i / num_points * max_action_dist, max_action_dist)
            print(f"Current max_dist: {max_dist}")  # Debug print
        
        if max_dist > 0.5:  # Minimum distance threshold
            print(f"Adding action: dist={max_dist}, angle={np.rad2deg(theta_rad):.2f}")  # Debug print
            actions.append((max_dist, theta_rad))
        else:
            print(f"Rejecting action: max_dist {max_dist} too small")  # Debug print

    print(f"\nFinal number of actions: {len(actions)}")  # Debug print
    
    # Draw actions on image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create a colored mask (green for navigable areas)
    colored_mask = np.zeros_like(img)
    colored_mask[navigability_mask] = [0, 255, 0]  # Green color for navigable areas
    
    # Blend the original image with the colored mask
    alpha = 0.5  # Transparency factor
    blended = cv2.addWeighted(img, 1, colored_mask, alpha, 0)
    
    # Draw arrows for actions
    for i, (dist, theta) in enumerate(actions):
        # Calculate arrow start point (below the image)
        start_y = resolution[0] + 50  # 50 pixels below the image
        start_x = center_px[0]  # Horizontally centered
        
        # Convert real-world distance to pixels using focal length
        # dist is in meters, need to convert to pixels
        pixel_dist = int(dist * focal_length)
        
        # Calculate arrow endpoint using the distance and angle
        end_x = start_x + int(pixel_dist * np.sin(theta))
        end_y = start_y - int(pixel_dist * np.cos(theta))
        
        # Find intersections with image boundaries
        intersections = find_intersections(start_x, start_y, end_x, end_y, resolution[1], resolution[0])
        
        if intersections:
            # Use only the intersection point that's inside or on the image boundary
            for point in intersections:
                if point[1] <= resolution[0]:  # Only use point if it's not below image
                    end_point = point
                    break
            else:
                continue  # Skip if no valid intersection found
            
            # Draw arrow only for the visible portion
            cv2.arrowedLine(blended, 
                           (int(start_x), int(resolution[0])), # Start from bottom edge
                           (int(end_point[0]), int(end_point[1])),
                           (255, 0, 0), 2, tipLength=0.3)
            
            # Add action number and distance near the arrow tip
            label = f"{i+1} ({dist:.1f}m)"
            cv2.putText(blended, label, 
                       (int(end_point[0]) + 10, int(end_point[1]) + 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Save visualization
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(blended)
    plt.title('Action Proposals with Navigation Mask')
    plt.axis('off')
    
    plt.savefig(output_path)
    plt.close()
    
    print(f"Action proposal visualization saved to {output_path}")
    return actions

if __name__ == "__main__":
    input_image = "./MVIMG_20250401_182609.jpg"
    output_image = "./indoor_actions.jpg"
    segmentation_model_path = "/data3/xu_ruochen/my_checkpoints/mask2former-swin-small-ade-semantic"
    actions = generate_action_proposals(input_image, output_image, segmentation_model_path)
    print(actions)
