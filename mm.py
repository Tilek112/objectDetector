import cv2
import tkinter as tk
from tkinter import filedialog, ttk
from concurrent.futures import ThreadPoolExecutor

from torch.backends import cudnn
from ultralytics import YOLO
from PIL import Image, ImageTk
import os
import shutil


# Function to process frames
def process_frame(frame):
    # Downsize the frame
    frame_resized = cv2.resize(frame, (1280, 720))  # Adjust the dimensions as per your model's input size

    # detect objects and track them
    results = openvino_model.track(frame_resized, persist=True)

    # plot results
    frame_processed = results[0].plot()

    return frame_processed


def load_video():
    global cap
    filename = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4")])
    if filename:
        cap = cv2.VideoCapture(filename)


# Set device to CUDA if available
device = "cuda"

# Initialize YOLO model
# openvino_model = YOLO("yolov8n_openvino_model/", task="detect").to(device)
openvino_model = YOLO("yolov8m.pt/", task="detect").to(device)
cudnn.benchmark = True  # Improve inference speed

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
canvas = tk.Canvas(root, width=1280, height=720)
canvas.pack()

cap = None

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

    root.after(25, play_video)

executor = ThreadPoolExecutor(max_workers=4)  # Adjust the number of workers as per your system

frames = []
play_video()

root.mainloop()
