import os
import time
import cv2
import struct
import matplotlib.pyplot as plt
import speech_recognition as sr
from serial.tools import list_ports
from ultralytics import YOLOWorld
from pydobot import Dobot
import serial.tools.list_ports

# ----------------------------
# Speech Recognition
# ----------------------------
r = sr.Recognizer()
with sr.Microphone() as source:
    print(" Say the object to detect (e.g., hammer, raspberry pi, arduino)...")
    audio = r.listen(source)

try:
    command = r.recognize_google(audio).lower()
    print(f" You said: {command}")
except sr.UnknownValueError:
    print(" Could not understand audio.")
    exit()

# ----------------------------
# Map speech to class_list
# ----------------------------
object_map = {
    "hammer": "metal hammer",
    "raspberry": "raspberry pi board",
    "arduino": "arduino uno board",
    "battery": "black battery pack",
    "cup": "paper cup",
    "ear buds": "white ear buds"
}

target_class = next((v for k, v in object_map.items() if k in command), None)
if target_class is None:
    print(" No matching object found in the speech.")
    exit()

print(f"🔍 Looking for: {target_class}")

# ----------------------------
# Initialize YOLOWorld
# ----------------------------
model_path = "yolov8s-worldv2.pt"
model = YOLOWorld(model_path)
model.set_classes([target_class])  # Only look for spoken object

# ----------------------------
# Open camera and detect
# ----------------------------
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print(" Error: Could not open camera.")
    exit()

plt.ion()
pixel_x = None
pixel_y = None

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, conf=0.01, iou=0.01, max_det=100, verbose=False)
        annotated_frame = results[0].plot()

        for box, cls_id in zip(results[0].boxes, results[0].boxes.cls):
            label = model.names[int(cls_id)]
            if label == target_class:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                pixel_x = int((x1 + x2) / 2)
                pixel_y = int((y1 + y2) / 2)

                cv2.putText(annotated_frame, f"({pixel_x}, {pixel_y})", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                print(f" Detected {label} at pixel: ({pixel_x}, {pixel_y})")
                raise StopIteration  # Break loop

        img_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        plt.axis('off')
        plt.pause(0.001)
        plt.clf()

except StopIteration:
    pass
finally:
    cap.release()
    plt.ioff()
    plt.close()

# ----------------------------
# Pixel to Robot Coordinates
# ----------------------------
if pixel_x is None or pixel_y is None:
    print(" Target object not found.")
    exit()
# Regression
robot_x = 0.7057 * pixel_x - 23.869
robot_y = -1.2439 * pixel_y + 302.6
print(f" Mapped Robot Coordinates: X={robot_x:.2f}, Y={robot_y:.2f}")

# ----------------------------
# Connect to Dobot and Move
# ----------------------------

available_ports = list(serial.tools.list_ports.comports())   # Detect available ports
port = None
for p in available_ports:
    if 'USB Serial Device' in p.description or 'CP210x' in p.description:
        port = p.device
        break

if not port:
    print(" Dobot not found on any COM port.")
    exit()

print(f" Connecting to Dobot on {port}...")
device = Dobot(port=port, verbose=True)
time.sleep(2)

# Clear queue
device._set_queued_cmd_clear()
time.sleep(1)

# Move to object (z and r kept same as initial)
pose = device.pose()
x, y, z, r = pose[:4]
print(f" Current Pose - x:{x:.1f}, y:{y:.1f}, z:{z:.1f}")

# Move to detected object's mapped coordinates
device.move_to(robot_x, robot_y, z, r, wait=True)
print(" Movement complete.")

device.close()
print(" Dobot disconnected.")
