import cv2
import tkinter as tk
from tkinter import filedialog, ttk
from concurrent.futures import ThreadPoolExecutor

import torch
from torch.backends import cudnn
from ultralytics import YOLO
from PIL import Image, ImageTk
import os
import shutil
import threading
import time
import numpy as np

# Set device to CUDA if available
device = "cuda"

# Initialize YOLO model
openvino_model = YOLO("yolov8m.pt/", task="detect").to(device)
cudnn.benchmark = True  # Improve inference speed

cap = None

# Function to load video
def load_video():
    global cap
    filename = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4")])
    if filename:
        cap = cv2.VideoCapture(filename)
        # Simulate a long-running process
        threading.Thread(target=simulate_long_running_process).start()

# Function to process frames
# Function to process frames
def process_frame(frame):
    # Downsize the frame
    frame_resized = cv2.resize(frame, (1920, 1080))  # Adjust the dimensions as per your model's input size
    # detect objects and track them
    results = openvino_model.track(frame_resized, persist=True)

    # Draw the detected objects on the frame
    for obj in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = [int(e) for e in obj]
        cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame_resized, f'{openvino_model.names[int(cls)]} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return frame_resized


# Function to apply some image processing
def apply_filter(frame):
    # Apply some simple filter, for example, grayscale
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Function to add timestamp to frames
def add_timestamp(frame):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    return frame

# Function to merge frames horizontally
def merge_frames_horizontal(frame1, frame2):
    # If any of the frames doesn't have the expected shape, return the other frame
    if len(frame1.shape) != 3 or len(frame2.shape) != 3:
        return frame1 if len(frame1.shape) == 3 else frame2

    # If the frames have different number of channels, convert one of them to match
    if frame1.shape[2] != frame2.shape[2]:
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_GRAY2BGR) if len(frame2.shape) == 2 else cv2.cvtColor(frame2,
                                                                                                      cv2.COLOR_RGB2BGR)

    return np.hstack((frame1, frame2))

# Function to save video
def save_video(frames):
    filename = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("Video files", "*.mp4")])
    if filename:
        temp_folder = "temp_frames"
        os.makedirs(temp_folder, exist_ok=True)

        for i, frame in enumerate(frames):
            cv2.imwrite(os.path.join(temp_folder, f"{i:04d}.png"), cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, 30, (1920, 1080))

        for i in range(len(frames)):
            frame = cv2.imread(os.path.join(temp_folder, f"{i:04d}.png"))
            out.write(frame)

        out.release()
        shutil.rmtree(temp_folder)

# Create GUI window
root = tk.Tk()
root.title("Object detector by Tilek")

# Style for the buttons
style = ttk.Style()
style.configure("TButton", padding=6, relief="raised")

# Create a frame for buttons
button_frame = tk.Frame(root)
button_frame.pack(pady=10)

# Load video button
load_button = ttk.Button(button_frame, text="Load Video", command=load_video)
load_button.grid(row=0, column=0, padx=10)

# Create a canvas to display video
canvas = tk.Canvas(root, width=1920, height=1080)
canvas.pack()

# Function to create a label for displaying information
def create_info_label():
    info_label = tk.Label(root, text="Waiting for video to load...", font=("Helvetica", 12))
    info_label.pack()

# Function to create a progress bar
def create_progress_bar():
    progress = ttk.Progressbar(root, orient="horizontal", length=200, mode="determinate")
    progress.pack()

    for i in range(101):
        progress['value'] = i
        root.update_idletasks()
        time.sleep(0.01)

    progress.destroy()

# Function to simulate long-running process
def simulate_long_running_process():
    time.sleep(5)  # Simulate a process that takes time
    create_progress_bar()

def play_video():
    global cap
    if cap is not None and cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Submit the frame for processing
            future = executor.submit(process_frame, frame)

            # Get the processed frame
            frame_processed = future.result()

            # Convert processed frame to a format compatible with Tkinter
            frame_processed_rgb = cv2.cvtColor(frame_processed, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_processed_rgb)
            img_tk = ImageTk.PhotoImage(image=img)

            # Display the frame
            canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
            canvas.img_tk = img_tk
        else:
            # Video has ended, show the "Save Video" button
            save_button = ttk.Button(button_frame, text="Save Video", command=lambda: save_video(frames))
            save_button.grid(row=0, column=1, padx=10)

    root.after(25, play_video)


executor = ThreadPoolExecutor(max_workers=4)  # Adjust the number of workers as per your system

frames = []
play_video()

root.mainloop()
