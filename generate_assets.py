import cv2
import numpy as np
import os

# --- SETUP ---
# Define the folders we need
FOLDERS = {
    "Assets": ["Hello", "Thanks", "ILoveYou"],
    "Assets/Alphabet": list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
}

def create_dummy_video(text, folder_path, filename):
    """Creates a 1-second video with the text written on it"""
    path = os.path.join(folder_path, f"{filename}.mp4")
    print(f"Creating: {path} ...")
    
    # Video settings
    width, height = 640, 480
    fps = 30
    seconds = 2 # Duration of clip
    
    # Initialize Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))
    
    # Create a black frame
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add Text to the frame (Center it)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    font_thickness = 3
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_x = (width - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2
    
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 255, 0), font_thickness)
    cv2.putText(frame, "(Placeholder Video)", (180, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1)
    
    # Write the same frame 60 times (2 seconds)
    for _ in range(fps * seconds):
        out.write(frame)
        
    out.release()

# --- MAIN LOOP ---
if __name__ == "__main__":
    for folder, items in FOLDERS.items():
        # Create folder if not exists
        if not os.path.exists(folder):
            os.makedirs(folder)
            
        # Generate videos
        for item in items:
            create_dummy_video(item, folder, item)
            
    print("\n✅ SUCCESS! All assets generated.")
    print("You can now run 'streamlit run app.py' and it will work perfectly.")