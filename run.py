# run.py

import cv2
from ultralytics import YOLO

def main():
    # Load your custom-trained YOLOv8 model
    model = YOLO('best.pt')

    # Define the video source
    video_path = 'input_video.mp4'
    cap = cv2.VideoCapture(video_path)

    # Get video properties for the output file
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create VideoWriter object
    output_path = 'output_video.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Or 'XVID'
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    print("Processing video... Press 'q' in the video window to quit early.")

    # Process video frame by frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection
        results = model(frame, verbose=False) # verbose=False hides console logs
        annotated_frame = results[0].plot() # Draws boxes on the frame

        # --- Compliance Logic ---
        # Get the names of detected objects
        detected_classes = [model.names[int(c)] for c in results[0].boxes.cls]
        
        # Check for compliance
        mask_found = 'Mask' in detected_classes
        gloves_found = 'Protective Gloves' in detected_classes
        # Using 'Goggles' as a stand-in for a head covering
        head_gear_found = 'Goggles' in detected_classes

        # --- Display Indicators ---
        def draw_status(img, text, pos, is_compliant):
            color = (0, 255, 0) if is_compliant else (0, 0, 255) # Green or Red
            cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        draw_status(annotated_frame, f"Mask: {'YES' if mask_found else 'NO'}", (10, 30), mask_found)
        draw_status(annotated_frame, f"Gloves: {'YES' if gloves_found else 'NO'}", (10, 60), gloves_found)
        draw_status(annotated_frame, f"Head Gear: {'YES' if head_gear_found else 'NO'}", (10, 90), head_gear_found)

        # Show the frame and save it
        cv2.imshow('FoodVision Hygiene Compliance', annotated_frame)
        out.write(annotated_frame)

        # Allow user to quit by pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"✅ Processing complete. Video saved as {output_path}")

if __name__ == "__main__":
    main()