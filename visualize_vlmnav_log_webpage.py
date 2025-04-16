import streamlit as st
import json
import os
import pandas as pd
from PIL import Image
import datetime

# Set page config
st.set_page_config(
    page_title="VLMNav Results Visualization",
    page_icon="🗺️",
    layout="wide"
)

# Constants
LOG_DIR = "/data3/xu_ruochen/vlmnav_vlm_result_log"
LOG_FILE = os.path.join(LOG_DIR, "vlmnav_results.jsonl")

def load_log_data():
    """Load and parse the JSONL log file."""
    data = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    # Convert timestamp string to datetime object
                    entry['datetime'] = datetime.datetime.strptime(
                        entry['timestamp'], 
                        "%Y%m%d_%H%M%S_%f"
                    )
                    data.append(entry)
                except json.JSONDecodeError:
                    st.warning(f"Skipped invalid JSON line: {line}")
    return data

def main():
    st.title("🗺️ VLMNav Results Visualization")
    
    # Load data
    data = load_log_data()
    if not data:
        st.error("No log data found. Please make sure the log file exists and contains data.")
        return
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(data)
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Episode filter if episode_id exists
    if 'episode_id' in df.columns:
        episodes = sorted(df['episode_id'].unique())
        selected_episode = st.sidebar.selectbox(
            "Select Episode",
            ["All"] + list(episodes)
        )
    
    # Date filter
    dates = sorted(df['datetime'].dt.date.unique())
    selected_date = st.sidebar.selectbox(
        "Select Date",
        ["All"] + list(dates)
    )
    
    # Apply filters
    filtered_df = df.copy()
    if selected_episode != "All" and 'episode_id' in df.columns:
        filtered_df = filtered_df[filtered_df['episode_id'] == selected_episode]
    if selected_date != "All":
        filtered_df = filtered_df[filtered_df['datetime'].dt.date == selected_date]
    
    # Display statistics
    st.header("📊 Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Entries", len(filtered_df))
    with col2:
        if 'episode_id' in filtered_df.columns:
            st.metric("Total Episodes", filtered_df['episode_id'].nunique())
    with col3:
        st.metric("Date Range", f"{filtered_df['datetime'].min().date()} to {filtered_df['datetime'].max().date()}")
    
    # Display results
    st.header("🖼️ Navigation Results")
    
    # Create columns for layout
    for idx, row in filtered_df.iterrows():
        with st.expander(f"Step {row.get('step_id', idx)} - {row['datetime']}"):
            col1, col2 = st.columns([1, 1])
            
            # Display image
            with col1:
                if os.path.exists(row['image_path']):
                    image = Image.open(row['image_path'])
                    st.image(image, caption="Navigation View", use_column_width=True)
                else:
                    st.error("Image file not found")
            
            # Display metadata
            with col2:
                st.subheader("Navigation Details")
                st.write("**VLM Output:**")
                st.write(row['vlm_output'])
                st.write("**Selected Action:**", row['action_number'])
                
                if 'additional_info' in row:
                    st.write("**Additional Information:**")
                    st.json(row['additional_info'])
                
                if 'episode_id' in row:
                    st.write("**Episode ID:**", row['episode_id'])
                if 'step_id' in row:
                    st.write("**Step ID:**", row['step_id'])

if __name__ == "__main__":
    main()
