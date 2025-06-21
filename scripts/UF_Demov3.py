import os
import cv2
import time
import math
import matplotlib.pyplot as plt
import speech_recognition as sr
from ultralytics import YOLOWorld
from xarm.wrapper import XArmAPI

# ----------------------------
# 1. Speech Recognition
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
# 2. Map speech to class_list
# ----------------------------
object_map = {
    "hammer": "metal hammer",
    "raspberry": "raspberry pi board",
    "arduino": "arduino uno board",
    "battery": "black battery pack",
    "cup": "paper cup",
    "ear buds": "white ear buds",
    "blue cube": "blue cube solid"
}
target_class = next((v for k, v in object_map.items() if k in command), None)
if not target_class:
    print(" No matching object found in map.")
    exit()

print(f"🔍 Looking for: {target_class}")

# ----------------------------
# 3. Load YOLOWorld model
# ----------------------------
model_path = "yolov8s-worldv2.pt"
model = YOLOWorld(model_path)
model.set_classes([target_class])

# ----------------------------
# 4. Open Webcam
# ----------------------------
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print(" Could not open camera.")
    exit()

plt.ion()
pixel_x, pixel_y = None, None

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, conf=0.002, iou=0.2, max_det=5, verbose=False)
        annotated_frame = results[0].plot()

        for box, cls_id in zip(results[0].boxes, results[0].boxes.cls):
            label = model.names[int(cls_id)]
            if label == target_class:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                pixel_x = int((x1 + x2) / 2)
                pixel_y = int((y1 + y2) / 2)

                cv2.putText(annotated_frame, f"{label} ({pixel_x}, {pixel_y})", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                print(f"📦 Detected {label} at ({pixel_x}, {pixel_y})")
                raise StopIteration  # Exit detection loop

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
# 5. Regression Mapping
# ----------------------------
if pixel_x is None or pixel_y is None:
    print(" Target object not detected.")
    exit()

# Replace with your regression coefficients
robot_x = -1.2029 * pixel_x + 25.501
robot_y = -0.2607 * pixel_y + 350.03


robot_z = 110  # Choose a safe Z height for pick
print(f"📍 Mapped Robot Coordinates: X={robot_x:.1f}, Y={robot_y:.1f}, Z={robot_z:.1f}")

# ----------------------------
# 6. Connect and Move xArm6
# ----------------------------
ip = "192.168.1.194"  # UPDATE with your xArm6 IP address
arm = XArmAPI(ip)
arm.motion_enable(enable=True)
arm.set_mode(0)
arm.set_state(state=0)

# # Optional: Move home
# arm.move_gohome(wait=True)

# Move above object
speed = 40
arm.set_position(x=robot_x, y=robot_y, z=robot_z, roll=180, pitch=0, yaw=0, speed=speed, wait=False)

# Lower down and grip
arm.set_position(x=robot_x, y=robot_y, z=robot_z, roll=180, pitch=0, yaw=0, speed=speed, wait=True)
arm.set_gripper_position(pos=0, wait=True)  # Close gripper (0 = fully closed)

# # Raise up
# arm.set_position(x=robot_x, y=robot_y, z=robot_z, roll=180, pitch=0, yaw=0, speed=speed, wait=True)

# # Move to drop position
# drop_x, drop_y, drop_z = 170, 170, robot_z
# arm.set_position(x=drop_x, y=drop_y, z=drop_z, roll=180, pitch=0, yaw=0, speed=speed, wait=True)
# arm.set_position(x=drop_x, y=drop_y, z=drop_z - 50, roll=180, pitch=0, yaw=0, speed=speed, wait=True)

# # Release object
# arm.set_gripper_position(pos=850, wait=True)  # Open gripper (850 = fully open)

# # Go back up
# arm.set_position(x=drop_x, y=drop_y, z=drop_z, roll=180, pitch=0, yaw=0, speed=speed, wait=True)

# # Optional: Go home and disconnect
# arm.move_gohome(wait=True)
arm.disconnect()

print("✅ Task complete. xArm6 disconnected.")
