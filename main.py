import cv2
import os
import numpy as np
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

# Load YOLO
def load_yolo(weights_path, cfg_path):
    print("Loading YOLO")
    net = cv2.dnn.readNet(weights_path, cfg_path)
    layer_names = net.getLayerNames()
    print("Layer Names:", layer_names)
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    return net, output_layers

def load_classes(names_path):
    with open(names_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    return classes

# Detect objects in a frame
def detect_objects(net, output_layers, frame):
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    return outs, width, height

# Get bounding boxes and class labels
def get_bounding_boxes(outs, width, height, conf_threshold=0.5, nms_threshold=0.4):
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    return boxes, confidences, class_ids, indexes

def save_cropped_objects(frame, boxes, indexes, output_folder):
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            cropped_img = frame[y:y+h, x:x+w]
            if cropped_img.size == 0:
                print(f"Warning: Cropped image for index {i} is empty.")
                continue
            output_path = os.path.join(output_folder, f"object_{i}.jpg")
            cv2.imwrite(output_path, cropped_img)


# Process the video
def process_video(video_path, output_folder, weights_path, cfg_path, names_path):
    net, output_layers = load_yolo(weights_path, cfg_path)
    classes = load_classes(names_path)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        outs, width, height = detect_objects(net, output_layers, frame)
        boxes, confidences, class_ids, indexes = get_bounding_boxes(outs, width, height)
        save_cropped_objects(frame, boxes, indexes, output_folder)
    cap.release()
    cv2.destroyAllWindows()

# Upload to Google Drive
def upload_to_drive(local_path, drive_folder_id):
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()
    drive = GoogleDrive(gauth)
    
    for filename in os.listdir(local_path):
        file_path = os.path.join(local_path, filename)
        if os.path.isfile(file_path):
            gfile = drive.CreateFile({'parents': [{'id': drive_folder_id}]})
            gfile.SetContentFile(file_path)
            gfile.Upload()
            print(f"Uploaded {filename} to Google Drive")

# Main execution
if __name__ == "__main__":
    video_path = 'nature.mp4'
    output_folder = 'output_objects'
    weights_path = 'yolov3.weights'  
    cfg_path = 'yolov3.cfg'         
    names_path = 'coco.names'        
    
    process_video(video_path, output_folder, weights_path, cfg_path, names_path)
    
    drive_folder_id = '1TCLXiLUmLvLVOb5GYE03-gcnqWw2P8OT' 
    upload_to_drive(output_folder, drive_folder_id)
