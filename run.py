# run.py (Final Multi-Video Processing Version)

import cv2
from ultralytics import YOLO
import argparse  # Used to read file names from the terminal
import os        # Used for handling file paths

def process_video(input_video_path):
    """
    This function takes a single video file path, processes it,
    and saves the output.
    """
    model = YOLO('best.pt')
    
    # Check if the input video file exists
    if not os.path.exists(input_video_path):
        print(f"--- ⚠️  Skipping: Input video not found at '{input_video_path}' ---")
        return

    cap = cv2.VideoCapture(input_video_path)

    # Check if the video opened successfully
    if not cap.isOpened():
        print(f"--- ⚠️  Skipping: Could not open video file at '{input_video_path}' ---")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Automatically create a unique name for the output file
    base_name = os.path.splitext(os.path.basename(input_video_path))[0]
    output_path = f"{base_name}_output.avi"
    
    # Define the codec for the output video file
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    if not out.isOpened():
        print(f"--- ⚠️  Skipping: Could not create output file '{output_path}' ---")
        cap.release()
        return
        
    # --- Flicker Reduction Setup ---
    # The status will only change if an object is missing for this many consecutive frames.
    PERSISTENCE_THRESHOLD = 15 
    mask_counter = 0
    gloves_counter = 0

    print(f"➡️  Processing '{input_video_path}'...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run the model on the frame with a low confidence to catch weaker detections
        # results = model(frame, verbose=False, conf=0.20)
        results = model(frame, verbose=False, conf=0.50) # Increased threshold to 50% 
        
        # Draw the bounding boxes on the frame
        annotated_frame = results[0].plot()

        # Get the list of all detected classes in the frame
        detected_classes = [model.names[int(c)] for c in results[0].boxes.cls]
        
        is_mask_in_frame = 'mask' in detected_classes
        is_gloves_in_frame = 'Gloves' in detected_classes

        # Update the flicker-reduction counters
        if is_mask_in_frame:
            mask_counter = PERSISTENCE_THRESHOLD
        else:
            mask_counter = max(0, mask_counter - 1)

        if is_gloves_in_frame:
            gloves_counter = PERSISTENCE_THRESHOLD
        else:
            gloves_counter = max(0, gloves_counter - 1)
            
        # Determine the final status based on the counters
        mask_found = mask_counter > 0
        gloves_found = gloves_counter > 0

        # --- Function to draw the status text on the screen ---
        def draw_status(img, text, y_pos, is_compliant, frame_w):
            color = (0, 255, 0) if is_compliant else (0, 0, 255) # Green for YES, Red for NO
            font_scale = 5.0  # Controls text size
            thickness = 8     # Controls text boldness
            
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            
            # Position text at the top right, with padding
            text_x = frame_w - text_width - 260 # Padding from the right edge
            text_y = y_pos                      # Padding from the top edge

            cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

        # Draw the "Mask" and "Gloves" status
        draw_status(annotated_frame, f"Mask: {'YES' if mask_found else 'NO'}", 190, mask_found, frame_width)
        draw_status(annotated_frame, f"Gloves: {'YES' if gloves_found else 'NO'}", 300, gloves_found, frame_width)

        # Write the processed frame to the output video file
        out.write(annotated_frame)

    # Clean up and release all resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"✅ Finished. Video saved as '{output_path}'")

if __name__ == "__main__":
    # This section handles reading one or more file names from the terminal
    parser = argparse.ArgumentParser(description="Process one or more videos to detect hygiene compliance.")
    parser.add_argument("video_files", type=str, nargs='+', help="Path(s) to the input video file(s).")
    args = parser.parse_args()
    
    # Loop through each video file provided and process it
    for video_file in args.video_files:
        process_video(video_file)
    
    print("\n🎉 All videos processed!")