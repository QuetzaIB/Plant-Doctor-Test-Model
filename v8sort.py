from pathlib import Path
import numpy as np
import cv2
from ultralytics import YOLO
import os

import deep_sort.deep_sort.deep_sort as ds


def putTextWithBackground(img, text, origin, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, text_color=(255, 255, 255), bg_color=(0, 0, 0), thickness=1):
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    bottom_left = origin
    top_right = (origin[0] + text_width, origin[1] - text_height - 5)
    cv2.rectangle(img, bottom_left, top_right, bg_color, -1)
    text_origin = (origin[0], origin[1] - 5)
    cv2.putText(img, text, text_origin, font, font_scale, text_color, thickness, lineType=cv2.LINE_AA)


def extract_detections(results, detect_class):
    detections = np.empty((0, 4))
    confarray = []

    for r in results:
        for box in r.boxes:
            if box.cls[0].int() == detect_class:
                x1, y1, x2, y2 = box.xywh[0].int().tolist()
                conf = round(box.conf[0].item(), 2)
                detections = np.vstack((detections, np.array([x1, y1, x2, y2])))
                confarray.append(conf)
    return detections, confarray


def detect_and_track(input_path: str, output_path: str, detect_class: int, model, tracker) -> Path:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error opening video file {input_path}")
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    output_video_path = Path(output_path) / "output.avi"

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    output_video = cv2.VideoWriter(output_video_path.as_posix(), fourcc, fps, size, isColor=True)
    frame_count = 0
    while True:
        success, frame = cap.read()

        if not (success):
            break
        frame_count += 1

        # Use the YoloV8 model to perform object detection on the current frame.
        results = model(frame, False, conf=0.9)

        # Extract detection information from the prediction results.
        detections, confarray = extract_detections(results, detect_class)

        # Use the DeepSORT model to track the detected objects.
        resultsTracker = tracker.update(detections, confarray, frame)

        for x1, y1, x2, y2, Id in resultsTracker:

            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])  # Convert the position information to integers.

            output_dir = './sort'
            # Create a corresponding ID folder.
            id_folder = os.path.join(output_dir, str(Id))
            os.makedirs(id_folder, exist_ok=True)

            image_count = len(os.listdir(id_folder))

            # Crop the image.
            cropped_image = frame[y1:y2, x1:x2]


            # Generate the filename.
            cropped_image_filename = os.path.join(id_folder, f"{frame_count}_{image_count + 1}.jpg")

            # Save the cropped image.
            cv2.imwrite(cropped_image_filename, cropped_image)

        for x1, y1, x2, y2, Id in resultsTracker:

            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2]) # Convert the position information to integers.

            # Draw bounding boxes and text.
            cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 0, 0), 3)
            putTextWithBackground(frame, str(int(Id)), (max(-10, x1), max(40, y1)), font_scale=1.5,
                                  text_color=(255, 255, 255), bg_color=(128, 0, 0), thickness=3)

        output_video.write(frame)

    output_video.release()
    cap.release()
    
    print(f'output dir is: {output_video_path}')
    return output_video_path


def track_cut_video(input_path = "demo.mp4", output_path = '.'):

    model = YOLO("best.pt")

    detect_class = 0
    print(f"detecting {model.names[detect_class]}")

    tracker = ds.DeepSort("deep_sort/deep_sort/deep/checkpoint/ckpt.t7")

    detect_and_track(input_path, output_path, detect_class, model, tracker)

if __name__ == "__main__":
    track_cut_video()
