from concurrent.futures import Future, ThreadPoolExecutor
import time
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
import tkinter as tk

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageTk

from network import NeuralNetwork


MODEL_PATH = Path(__file__).parent / "Saved_paramaters" / "784_128_64_10.npz"


def clean_foreground(image: np.ndarray) -> np.ndarray:
    """Extract dark writing from a grayscale image and remove tiny noise."""
    if image.ndim != 2 or image.size == 0:
        raise ValueError("Expected a non-empty grayscale image")

    height, width = image.shape
    scale = min(1.0, 512 / max(height, width))
    if scale < 1:
        image = cv2.resize(
            image,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )

    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    _, foreground = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )
    _, labels, stats, _ = cv2.connectedComponentsWithStats(
        foreground, connectivity=8
    )
    cleaned = np.zeros_like(foreground)
    min_area = max(3, int(foreground.size * 0.00001))
    keep_component = stats[:, cv2.CC_STAT_AREA] >= min_area
    keep_component[0] = False
    cleaned[keep_component[labels]] = 255
    return cleaned


def preprocess_digit(image: np.ndarray) -> np.ndarray | None:
    """Convert a grayscale image of one dark-on-light digit to an MNIST input."""
    cleaned = clean_foreground(image)
    ys, xs = np.nonzero(cleaned)
    if xs.size == 0:
        return None

    digit = cleaned[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    scale = 20 / max(digit.shape)
    width = max(1, round(digit.shape[1] * scale))
    height = max(1, round(digit.shape[0] * scale))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    digit = cv2.resize(digit, (width, height), interpolation=interpolation)

    canvas = np.zeros((28, 28), dtype=np.uint8)
    x_offset = (28 - width) // 2
    y_offset = (28 - height) // 2
    canvas[y_offset:y_offset + height, x_offset:x_offset + width] = digit

    moments = cv2.moments(canvas)
    if moments["m00"]:
        center_x = moments["m10"] / moments["m00"]
        center_y = moments["m01"] / moments["m00"]
        shift = np.float32([[1, 0, round(13.5 - center_x)],
                            [0, 1, round(13.5 - center_y)]])
        canvas = cv2.warpAffine(canvas, shift, (28, 28), flags=cv2.INTER_LINEAR)

    return (canvas.astype(np.float32) / 255).reshape(784, 1)


def segment_digits(image: np.ndarray) -> list[np.ndarray]:
    """Split separated handwritten digits into left-to-right grayscale crops."""
    height, width = image.shape
    scale = min(1.0, 512 / max(height, width))
    if scale < 1:
        image = cv2.resize(
            image,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    foreground = clean_foreground(image)
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(
        foreground, connectivity=8
    )
    components = [
        (x, y, width, height)
        for x, y, width, height, area in stats[1:component_count]
        if area >= 3
    ]
    if not components:
        return []

    median_height = float(np.median([height for _, _, _, height in components]))
    components = [
        component for component in components
        if component[3] >= max(4, median_height * 0.4)
    ]
    components.sort(key=lambda component: component[0])

    padding = max(2, round(median_height * 0.08))
    image_height, image_width = image.shape
    return [
        image[max(0, y - padding):min(image_height, y + height + padding),
              max(0, x - padding):min(image_width, x + width + padding)]
        for x, y, width, height in components
    ]


def format_digit_string(digits: str) -> str:
    """Group a recognized digit string in threes without discarding leading zeros."""
    if len(digits) <= 3:
        return digits
    first_group_length = len(digits) % 3 or 3
    groups = [digits[:first_group_length]]
    groups.extend(
        digits[index:index + 3]
        for index in range(first_group_length, len(digits), 3)
    )
    return ",".join(groups)


def camera_region(frame: np.ndarray) -> np.ndarray:
    """Return the central camera area as grayscale, matching model preprocessing."""
    height, width = frame.shape[:2]
    region = frame[int(height * 0.15):int(height * 0.85),
                   int(width * 0.20):int(width * 0.80)]
    return cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)


class CameraApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.model = NeuralNetwork.load(MODEL_PATH)
        if self.model.layer_sizes[0] != 784 or self.model.layer_sizes[-1] != 10:
            raise ValueError("The saved model must accept 784 inputs and return 10 classes")

        self.camera: cv2.VideoCapture | None = None
        self.latest_frame: np.ndarray | None = None
        self.loaded_image: np.ndarray | None = None
        self.refresh_id: str | None = None
        self.prediction_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="digit-prediction"
        )
        self.prediction_future: Future[str] | None = None
        self.prediction_poll_id: str | None = None
        self.last_live_prediction = 0.0
        self.live_prediction = tk.BooleanVar(value=False)
        self.multiple_digits = tk.BooleanVar(value=False)
        self.result = tk.StringVar(value="Start a camera or open an image to begin.")

        root.title("MNIST Handwritten Digit Recogniser")
        root.geometry("1000x720")
        root.minsize(700, 500)

        ttk.Label(
            root,
            text="Show one dark handwritten digit on plain white paper, centered in the guide.",
            wraplength=950,
        ).pack(padx=16, pady=(16, 8))

        controls = ttk.Frame(root)
        controls.pack(fill="x", padx=16, pady=8)
        ttk.Label(controls, text="Camera index:").pack(side="left")
        self.camera_index = tk.IntVar(value=0)
        tk.Spinbox(controls, from_=0, to=10, width=4, textvariable=self.camera_index).pack(
            side="left", padx=(6, 12)
        )
        self.camera_button = ttk.Button(
            controls, text="Start camera", command=self.toggle_camera
        )
        self.camera_button.pack(side="left", padx=4)
        ttk.Button(controls, text="Open image...", command=self.open_image).pack(
            side="left", padx=4
        )
        ttk.Button(
            controls, text="Capture / predict", command=self.capture_and_predict
        ).pack(side="left", padx=4)
        ttk.Checkbutton(
            controls, text="Live prediction", variable=self.live_prediction
        ).pack(side="left", padx=(12, 4))
        ttk.Checkbutton(
            controls, text="Multiple digits", variable=self.multiple_digits
        ).pack(side="left", padx=4)

        self.preview = ttk.Label(root, text="Camera or image preview", anchor="center")
        self.preview.pack(fill="both", expand=True, padx=16, pady=8)
        ttk.Label(
            root, textvariable=self.result, font=("Segoe UI", 18, "bold")
        ).pack(padx=16, pady=(8, 16))

        root.protocol("WM_DELETE_WINDOW", self.close)

    def toggle_camera(self) -> None:
        if self.camera is not None:
            self.stop_camera()
            return

        try:
            index = self.camera_index.get()
        except tk.TclError:
            messagebox.showerror("Invalid camera", "Enter a camera index from 0 to 10.")
            return

        camera = cv2.VideoCapture(index)
        if not camera.isOpened():
            camera.release()
            messagebox.showerror(
                "Camera unavailable",
                f"Could not open camera {index}. Check that it is connected and not in use.",
            )
            return

        self.camera = camera
        self.loaded_image = None
        self.latest_frame = None
        self.camera_button.configure(text="Stop camera")
        self.result.set(
            "Keep the digit(s) inside the guide, then capture or enable live prediction."
        )
        self.refresh_camera()

    def stop_camera(self) -> None:
        if self.refresh_id is not None:
            self.root.after_cancel(self.refresh_id)
            self.refresh_id = None
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        self.latest_frame = None
        self.camera_button.configure(text="Start camera")

    def refresh_camera(self) -> None:
        if self.camera is None:
            return
        success, frame = self.camera.read()
        if not success:
            self.stop_camera()
            messagebox.showerror(
                "Camera error", "The camera stopped providing video frames."
            )
            return

        self.latest_frame = frame
        self.show_camera_frame(frame)
        now = time.monotonic()
        if (
            self.live_prediction.get()
            and self.prediction_future is None
            and now - self.last_live_prediction >= 0.5
        ):
            self.start_prediction(camera_region(frame))
            self.last_live_prediction = now
        self.refresh_id = self.root.after(30, self.refresh_camera)

    def show_camera_frame(self, frame: np.ndarray) -> None:
        height, width = frame.shape[:2]
        scale = min(780 / width, 520 / height, 1)
        display_width = max(1, round(width * scale))
        display_height = max(1, round(height * scale))
        resized = cv2.resize(frame, (display_width, display_height))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        draw = ImageDraw.Draw(image)
        draw.rectangle(
            (int(display_width * 0.20), int(display_height * 0.15),
             int(display_width * 0.80), int(display_height * 0.85)),
            outline="lime",
            width=3,
        )
        self.show_image(image)

    def show_image(self, image: Image.Image) -> None:
        image.thumbnail((780, 520), Image.Resampling.LANCZOS)
        self.preview_photo = ImageTk.PhotoImage(image)
        self.preview.configure(image=self.preview_photo, text="")

    def open_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose an image containing one handwritten digit",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        self.stop_camera()
        try:
            with Image.open(path) as image:
                self.loaded_image = np.asarray(image.convert("L"))
                self.show_image(image.convert("RGB").copy())
        except OSError as error:
            self.loaded_image = None
            messagebox.showerror("Could not open image", str(error))
            return
        self.result.set("Image loaded. Click Capture / predict to classify it.")

    def capture_and_predict(self) -> None:
        if self.camera is not None and self.latest_frame is not None:
            self.start_prediction(camera_region(self.latest_frame))
        elif self.loaded_image is not None:
            self.start_prediction(self.loaded_image)
        else:
            self.result.set("Start a camera or open an image first.")

    def start_prediction(self, grayscale: np.ndarray) -> None:
        if self.prediction_future is not None:
            self.result.set("Prediction in progress; please wait.")
            return
        self.result.set("Predicting...")
        self.prediction_future = self.prediction_executor.submit(
            self.predict_image, grayscale.copy(), self.multiple_digits.get()
        )
        self.prediction_poll_id = self.root.after(50, self.poll_prediction)

    def poll_prediction(self) -> None:
        self.prediction_poll_id = None
        future = self.prediction_future
        if future is None:
            return
        if not future.done():
            self.prediction_poll_id = self.root.after(50, self.poll_prediction)
            return

        self.prediction_future = None
        try:
            self.result.set(future.result())
        except Exception as error:
            self.result.set(f"Prediction failed: {error}")

    def predict_image(self, grayscale: np.ndarray, multiple_digits: bool) -> str:
        if multiple_digits:
            return self.predict_multiple_digits(grayscale)

        digit = preprocess_digit(grayscale)
        if digit is None:
            return "No clear digit found. Try darker writing and better lighting."

        probabilities = self.model.forward_prop(digit)[:, 0]
        prediction = int(np.argmax(probabilities))
        confidence = float(probabilities[prediction]) * 100
        return f"Prediction: {prediction}  |  Confidence: {confidence:.1f}%"

    def predict_multiple_digits(self, grayscale: np.ndarray) -> str:
        crops = segment_digits(grayscale)
        if not crops:
            return "No clear digits found. Try darker writing and better lighting."

        inputs = [preprocess_digit(crop) for crop in crops]
        inputs = [digit for digit in inputs if digit is not None]
        if not inputs:
            return "No clear digits found. Try darker writing and better lighting."

        probabilities = self.model.forward_prop(np.hstack(inputs))
        predictions = np.argmax(probabilities, axis=0)
        digits = "".join(str(int(prediction)) for prediction in predictions)
        return f"Number: {format_digit_string(digits)}  |  Digits: {digits}"

    def close(self) -> None:
        self.stop_camera()
        if self.prediction_poll_id is not None:
            self.root.after_cancel(self.prediction_poll_id)
            self.prediction_poll_id = None
        self.prediction_executor.shutdown(wait=False, cancel_futures=True)
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    CameraApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
