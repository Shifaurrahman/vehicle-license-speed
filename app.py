import cv2
import numpy as np
from ultralytics import YOLO
from sort.sort import *
from util import get_car, read_license_plate, write_csv

class SpeedEstimator:
    def __init__(self):
        self.prev_positions = {}
    
    def calculate_speed(self, current_positions, fps):
        speeds = {}
        for car_id, current_pos in current_positions.items():
            if car_id in self.prev_positions:
                prev_pos = self.prev_positions[car_id]
                distance = np.linalg.norm(np.array(current_pos) - np.array(prev_pos))
                speed = (distance / fps) * 3.6  # Convert m/s to km/h
                speeds[car_id] = speed
        self.prev_positions = current_positions
        return speeds

# Initialize YOLO models
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('license_plate_detector.pt')

# Load video
input_video_path = './sample_video.mp4'
output_video_path = './output_video1.avi'
cap = cv2.VideoCapture(input_video_path)

# Video writer for output
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

results = {}
mot_tracker = Sort()
vehicles = [2, 3, 5, 7]  # Vehicle classes: car, motorcycle, bus, truck

speed_estimator = SpeedEstimator()

# Read frames
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        results[frame_nmr] = {}
        # Detect vehicles
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # Track vehicles
        track_ids = mot_tracker.update(np.asarray(detections_))

        # Collect current positions for speed calculation
        current_positions = {track_id[4]: ((track_id[0] + track_id[2]) / 2, (track_id[1] + track_id[3]) / 2) for track_id in track_ids}

        # Calculate speeds
        speeds = speed_estimator.calculate_speed(current_positions, fps)

        # Detect license plates
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # Assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            if car_id != -1:
                # Crop license plate
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                # Process license plate
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                # Read license plate number
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                if license_plate_text is not None:
                    speed = speeds.get(car_id, 0)
                    results[frame_nmr][car_id] = {
                        'car': {'bbox': [xcar1, ycar1, xcar2, ycar2], 'speed': speed},
                        'license_plate': {'bbox': [x1, y1, x2, y2], 'text': license_plate_text, 'bbox_score': score, 'text_score': license_plate_text_score}
                    }

                    # Annotate frame
                    cv2.rectangle(frame, (int(xcar1), int(ycar1)), (int(xcar2), int(ycar2)), (0, 255, 0), 2)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    cv2.putText(frame, license_plate_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.putText(frame, f'{speed:.2f} km/h', (int(xcar1), int(ycar1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Write frame to output video
        out.write(frame)

# Release resources
cap.release()
out.release()

# Write results to CSV file
write_csv(results, './test.csv')

print(f"Output video saved as: {output_video_path}")
print(f"CSV results saved as: ./test1.csv")
