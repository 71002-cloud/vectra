import cv2
import threading
import queue
import keyboard
from pathlib import Path

from detection_display_class import FrameCapture, Display

camera_index = [1, 2]
camera_amount = len(camera_index)
queue_size_2 = 2

class TakePictures:
    def __init__(self, capture, queue_size):
        self.frame_queue = queue.Queue(maxsize=queue_size)
        self.capture = capture

        self.thread = None
        self.running = False
        self.image_folder = (Path(__file__).resolve().parent / "calibration_images")
        self.left_folder = self.image_folder / "left_camera"
        self.right_folder = self.image_folder / "right_camera"
        self.images_taken = 0

        self.s_was_pressed = False

    def start(self):
        self.left_folder.mkdir(parents=True, exist_ok=True)
        self.right_folder.mkdir(parents=True, exist_ok=True)

        numbers = []

        for folder in (self.left_folder, self.right_folder):
            for file in folder.glob("*.png"):
                if file.stem.isdigit():
                    numbers.append(int(file.stem))

        self.images_taken = max(numbers, default=0)

        print(f"Existing highest image number: {self.images_taken}")

        self.running = True

        self.thread = threading.Thread(target=self._standby, daemon=True)
        self.thread.start()

        return self

    def take_picture(self, data):
        if len(data) != 2:
            print("Both camera frames are required.")
            return

        left_frame = data[0]["frame"].copy()
        right_frame = data[1]["frame"].copy()

        number = self.images_taken + 1

        while (self.left_folder / f"{number:03d}.png").exists() or (self.right_folder / f"{number:03d}.png").exists():
            number += 1

        left_path = self.left_folder / f"{number:03d}.png"
        right_path = self.right_folder / f"{number:03d}.png"

        left_saved = cv2.imwrite(str(left_path), left_frame)
        right_saved = cv2.imwrite(str(right_path), right_frame)

        if left_saved and right_saved:
            self.images_taken = number
            print(f"Saved image pair {number:03d}")
        else:
            if left_saved:
                left_path.unlink(missing_ok=True)

            if right_saved:
                right_path.unlink(missing_ok=True)

            print("Failed to save image pair.")

    def _standby(self):
        while self.running:
            data = self.capture.get_frame()

            if data is None:
                continue

            s_pressed = keyboard.is_pressed("s")

            if s_pressed and not self.s_was_pressed:
                self.take_picture(data)

            self.s_was_pressed = s_pressed

            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass

            try:
                self.frame_queue.put_nowait(data)
            except queue.Full:
                pass

    def get_frame(self, timeout=1.0):
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        self.running = False

        if self.thread is not None:
            self.thread.join(timeout=2.0)

class CalibrationApp:
    def __init__(self, capture, display, picture):
        self.capture = capture
        self.display = display
        self.picture = picture

    def start(self):
        self.capture.start()
        self.picture.start()
        self.display.start()

    def stop(self):
        self.capture.stop()
        self.picture.stop()
        self.display.stop()
        cv2.destroyAllWindows()

def main():
    capture = FrameCapture(camera_index=camera_index, camera_amount=camera_amount, queue_size=queue_size_2)
    picture = TakePictures(capture=capture, queue_size=queue_size_2)
    display = Display(last_frame_owner=picture, camera_amount=camera_amount)
    app = CalibrationApp(capture=capture, display=display, picture=picture)

    try:
        app.start()
        while display.running:
            display.display_loop()

    finally:
        app.stop()


if __name__ == "__main__":
    main()
