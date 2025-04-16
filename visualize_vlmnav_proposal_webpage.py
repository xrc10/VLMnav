import os
import json
import streamlit as st
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import glob
from datetime import datetime

# Directory where images are stored
IMAGE_DIR = "/data3/xu_ruochen/vlmnav_action_proposal_log"
LOGS_PATH = os.path.join(IMAGE_DIR, "logs.json")

st.set_page_config(
    page_title="VLMNav Action Proposal Visualization",
    page_icon="🧭",
    layout="wide"
)

def load_data():
    """Load all images and their corresponding action data."""
    # Get all jpg images in the directory
    image_files = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.jpg")))
    
    # Load action info from logs.json
    action_data = []
    if os.path.exists(LOGS_PATH):
        with open(LOGS_PATH, 'r') as f:
            for line in f:
                if line.strip():
                    action_data.append(json.loads(line.strip()))
    
    return image_files, action_data

def extract_timestamp(filename):
    """Extract timestamp from filename."""
    basename = os.path.basename(filename)
    timestamp = basename.split('.')[0]
    try:
        dt = datetime.strptime(timestamp, "%Y%m%d_%H%M%S_%f")
        return dt
    except:
        return datetime.min

def main():
    st.title("VLMNav Action Proposal Visualization")
    
    # Load data
    image_files, action_data = load_data()
    
    if not image_files:
        st.warning("No images found in the directory.")
        return
    
    # Sidebar for filtering and options
    st.sidebar.header("Options")
    
    # Sort by timestamp
    sort_order = st.sidebar.selectbox(
        "Sort Order",
        ["Newest First", "Oldest First"]
    )
    
    # Number of images to display per page
    items_per_page = st.sidebar.slider("Images per page", 1, 20, 10)
    
    # Sort the image files based on timestamp
    sorted_images = sorted(
        image_files, 
        key=extract_timestamp,
        reverse=(sort_order == "Newest First")
    )
    
    # Pagination
    total_pages = (len(sorted_images) + items_per_page - 1) // items_per_page
    if total_pages > 0:
        page_number = st.sidebar.number_input(
            f"Page (1-{total_pages})", 
            min_value=1, 
            max_value=total_pages, 
            value=1
        )
    else:
        page_number = 1
    
    start_idx = (page_number - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, len(sorted_images))
    
    # Display image count information
    st.sidebar.info(f"Showing {end_idx - start_idx} of {len(sorted_images)} images")
    
    # Display images for the current page
    for i, img_path in enumerate(sorted_images[start_idx:end_idx]):
        col1, col2 = st.columns([3, 2])
        
        # Extract timestamp from filename
        timestamp_str = os.path.basename(img_path).split('.')[0]
        
        with col1:
            st.subheader(f"Image {i+1 + start_idx}")
            # Display timestamp if available
            try:
                dt = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S_%f")
                st.caption(f"Timestamp: {dt.strftime('%Y-%m-%d %H:%M:%S.%f')}")
            except:
                st.caption("Timestamp not available")
            
            # Display the image
            img = Image.open(img_path)
            st.image(img, use_column_width=True)
        
        with col2:
            st.subheader("Action Proposals")
            
            # Find corresponding action data
            actions = None
            for action_entry in action_data:
                entry_timestamp = timestamp_str
                
                if i < len(action_data):
                    actions = action_data[i]
                    
                    # Create a table of action information
                    if actions:
                        action_df = pd.DataFrame(actions)
                        action_df = action_df.rename(columns={
                            'action_number': 'Action Number',
                            'turning_degree': 'Turning Degree (°)'
                        })
                        st.table(action_df.set_index('Action Number'))
                    else:
                        st.info("No action proposals available for this image")
                    break
            else:
                st.info("No action proposals found for this image")
        
        st.markdown("---")
    
    # Display summary statistics
    if action_data:
        st.sidebar.subheader("Summary Statistics")
        total_images = len(sorted_images)
        st.sidebar.info(f"Total Images: {total_images}")
        
        if len(action_data) > 0:
            # Calculate average actions per image
            actions_per_image = [len(actions) for actions in action_data if actions]
            if actions_per_image:
                avg_actions = sum(actions_per_image) / len(actions_per_image)
                st.sidebar.info(f"Average Actions per Image: {avg_actions:.2f}")

if __name__ == "__main__":
    main()
