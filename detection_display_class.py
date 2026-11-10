import cv2
from ultralytics import YOLO
from deepface import DeepFace
import time
import threading
import queue
import os
import numpy as np

class Person:
    def __init__(self, track_id, identity_id):
        self.track_id = track_id
        self.last_face_check = 0
        self.name = "Unknown"
        self.verified = False

        self.identity_id = identity_id
        self.appearance = None
        self.last_seen = time.time()

class PeopleManager:
    def __init__(self):
        self.people = {}
        self.track_to_person = {}
        self.next_identity_id = 1

    def get_person(self, track_id):
        return self.track_to_person.get(track_id)

    def update_person(self, track_id, appearance, current_tracks_ids):
        person = self.track_to_person.get(track_id)

        if person is not None:
            person.last_seen = time.time()
            person.track_id = track_id

            if appearance is not None:
                person.appearance = appearance

            return person

        best_person = None
        best_distance = float("inf")
        now = time.time()

        if appearance is not None:
            for candidate in list(self.people.values()):
                old_track_id = candidate.track_id

                if old_track_id in current_tracks_ids:
                    continue

                if now - candidate.last_seen > 30 or candidate.appearance is None:
                    self.people.pop(candidate.identity_id, None)
                    self.track_to_person.pop(old_track_id, None)
                    continue

                distance = cv2.compareHist(
                    candidate.appearance,
                    appearance,
                    cv2.HISTCMP_BHATTACHARYYA
                )

                if distance < best_distance:
                    best_distance = distance
                    best_person = candidate

        if best_person is not None and best_distance < 0.25:
            person = best_person
            old_track_id = person.track_id
            person.track_id = track_id
            person.last_seen = now
            person.appearance = appearance
            self.track_to_person[track_id] = person
            self.track_to_person.pop(old_track_id, None)

            print(f"Track_id: {track_id} - matches: {person.identity_id} - (distance={best_distance:.3f}")
            return person

        identity_id = self.next_identity_id
        self.next_identity_id += 1
        person = Person(track_id, identity_id)
        person.appearance = appearance
        person.last_seen = now

        self.people[identity_id] = person
        self.track_to_person[track_id] = person

        return person

    def update_verification(self, track_id, name, verified):
        person = self.get_person(track_id)

        if person is not None:
            self.track_to_person[track_id].name = name
            self.track_to_person[track_id].verified = verified
            self.people[person.identity_id].name = name
            self.people[person.identity_id].verified = verified

    def get_people(self):
        return self.people

class FaceDetector:
    def __init__(self, queue_size, photo_folder, people_manager):
        self.face_queue = queue.Queue(maxsize=queue_size)
        self.photo_folder = photo_folder
        self.people_manager = people_manager

        self.running = False
        self.thread = None
        self.reference_embeddings = {}

    def start(self):
        self.running = True
        self.set_reference_embeddings()

        self.thread = threading.Thread(target=self._face_loop, daemon=True)
        self.thread.start()

        return self

    def set_reference_embeddings(self):
        # Reference images are stored as "Name_1.jpg", "Name_2.jpg", etc.
        self.reference_embeddings.clear()

        for file in os.listdir(self.photo_folder):
            if not file.lower().endswith(".jpg"):
                continue

            name = file.rsplit("_", 1)[0]
            image_path = os.path.join(self.photo_folder, file)

            if name not in self.reference_embeddings:
                self.reference_embeddings[name] = []
                try:
                    result = DeepFace.represent(img_path=image_path, model_name="Facenet", detector_backend="opencv", enforce_detection=True)
                    embedding = result[0]["embedding"]
                    self.reference_embeddings[name].append(embedding)

                except ValueError as e:
                    print(f"Error occurred while processing {image_path}: {e}")
                    continue

    @staticmethod
    def compare_embeddings(embedding1, embedding2):
        denominator = np.linalg.norm(embedding1) * np.linalg.norm(embedding2)

        if denominator == 0:
            return float("inf")

        return float(1 - np.dot(embedding1, embedding2) / denominator)
    
    def find_best_match(self, detected_embedding, threshold=0.4):
        best_name = None
        best_distance = float("inf")

        for name, embeddings in self.reference_embeddings.items():
            for reference_embedding in embeddings:
                distance = self.compare_embeddings(detected_embedding, reference_embedding)

                if distance < best_distance:
                    best_distance = distance
                    best_name = name

        if best_distance < threshold:
            print(f"Best match: {best_name} with distance: {best_distance:.3f}")
            return best_name
        else:
            return None

    def _face_loop(self):
        while self.running:
            task = self.face_queue.get()

            if task is None:
                break

            track_id, face_crop = task

            try:
                result = DeepFace.represent(img_path=face_crop, model_name="Facenet", detector_backend="opencv", enforce_detection=True)
                detected_embedding = np.array(result[0]["embedding"], dtype=np.float32)

                name = self.find_best_match(detected_embedding)

                if name is not None:
                    name = name[0].upper() + name[1:].lower()
                    self.people_manager.update_verification(track_id, name, True)

            except Exception as error:
                print(f"Failed to perform face check for track {track_id}", error)
            
            finally:
                self.face_queue.task_done()

    def add_to_queue(self, track_id, face_crop):
        self.face_queue.put_nowait((track_id, face_crop))

    def stop(self):
        self.running = False

        try:
            self.face_queue.put_nowait(None)
        except queue.Full:
            pass

        if self.thread is not None:
            self.thread.join(timeout=2.0)

class FrameCapture:
    def __init__(self, camera_index, queue_size):
        self.camera_index = camera_index
        self.frame_queue = queue.Queue(maxsize=queue_size)

        self.camera = None
        self.thread = None
        self.running = False

    def start(self):
        self.camera = cv2.VideoCapture(self.camera_index)

        if not self.camera.isOpened():
            raise RuntimeError("Could not open camera")

        # Use 1280x720 to improve face detection without significantly affecting performance.
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.camera.set(cv2.CAP_PROP_FPS, 30)

        self.running = True

        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

        return self

    def _capture_loop(self):
        while self.running:
            success, frame = self.camera.read()

            if not success:
                continue

            frame_data = {
                "frame": frame,
                "timestamp": time.perf_counter()
            }

            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass

            try:
                self.frame_queue.put_nowait(frame_data)
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

        if self.camera is not None:
            self.camera.release()
            self.camera = None

class Detection:
    def __init__(self, conf, iou, queue_size, capture, people_manager, face_detector, appearance_refresh):
        self.conf = conf
        self.iou = iou
        self.frame_queue = queue.Queue(maxsize=queue_size)
        self.capture = capture
        self.people_manager = people_manager
        self.face_detector = face_detector
        self.appearance_refresh_interval = appearance_refresh
        self.appearance_frame_count = 0

        self.model = None
        self.running = False
        self.thread = None

    def start(self):
        self.model = YOLO("yolo11m-pose.engine")
        self.running = True
        self.thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.thread.start()

        return self

    @staticmethod
    def get_appearance(frame, x1, y1, x2, y2):
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return None

        height = crop.shape[0]
        # Focus on clothing by excluding most of the head and feet.
        crop = crop[int(height*0.2):int(height * 0.9), :]

        if crop.size == 0:
            return None

        # Represent the crop using a hue-saturation histogram.
        hsv = cv2.cvtColor(crop[::2, ::2], cv2.COLOR_BGR2HSV)

        histogram = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
        cv2.normalize(histogram, histogram)
        return histogram

    @staticmethod
    def get_face_crop_xxyy(frame, keypoints):
        h, w = frame.shape[:2]

        # YOLO pose keypoints.
        nose = keypoints[0]
        left_eye = keypoints[1]
        right_eye = keypoints[2]
        left_ear = keypoints[3]
        right_ear = keypoints[4]

        # Require the nose and both eyes to be inside the frame.
        points = [nose, left_eye, right_eye]

        if not all(0 < p[0] < w and 0 < p[1] < h for p in points):
            return None

        # Estimate face width using the eyes and ears.
        valid_ears = [
            p for p in [left_ear, right_ear]
            if 0 < p[0] < w and 0 < p[1] < h
        ]

        eye_distance = abs(left_eye[0] - right_eye[0])

        if eye_distance < 2:
            return None

        if len(valid_ears) == 2:
            face_width = abs(left_ear[0] - right_ear[0])
        else:
            face_width = eye_distance * 2.2

        # Use the midpoint between the eyes as the face's horizontal center.
        center_x = (left_eye[0] + right_eye[0]) / 2

        # Estimate the top of the head above the eyes.
        eye_y = (left_eye[1] + right_eye[1]) / 2
        face_height = face_width * 1.25

        x1 = int(center_x - face_width * 0.65)
        x2 = int(center_x + face_width * 0.65)
        y1 = int(eye_y - face_height * 0.65)
        y2 = int(eye_y + face_height * 0.65)

        # Keep the crop inside the frame.
        x1 = max(0, x1)
        x2 = min(w, x2)
        y1 = max(0, y1)
        y2 = min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        return x1, x2, y1, y2

    def _detection_loop(self):
        while self.running:
            data = self.capture.get_frame()
            if data is None:
                continue

            frame = data["frame"]
            self.appearance_frame_count += 1
            refresh_appearances = (
                self.appearance_frame_count >= self.appearance_refresh_interval
            )
            frame_height, frame_width = frame.shape[:2]
            items = []
            current_time = time.time()
            results = self.model.track(frame, persist=True, classes=[0], verbose=False, conf=self.conf, iou=self.iou)[0]

            if results.boxes.id is not None:
                track_ids = results.boxes.id.int().tolist()
                current_track_ids = set(track_ids)
                keypoints = None
                if results.keypoints is not None:
                    keypoints = results.keypoints.xy[:, :5].cpu().numpy()

                for box_index, (box, track_id) in enumerate(zip(results.boxes, track_ids)):

                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                    x1 = max(0, min(x1, frame_width))
                    x2 = max(0, min(x2, frame_width))
                    y1 = max(0, min(y1, frame_height))
                    y2 = max(0, min(y2, frame_height))

                    existing_person = self.people_manager.get_person(track_id)
                    should_refresh_appearance = existing_person is None or refresh_appearances
                    appearance = (
                        self.get_appearance(frame, x1, y1, x2, y2) if should_refresh_appearance
                        else None
                    )

                    person = self.people_manager.update_person(track_id, appearance, current_track_ids)

                    face_xxyy = None
                    if keypoints is not None:
                        face_xxyy = self.get_face_crop_xxyy(frame, keypoints[box_index])

                    fx1, fx2, fy1, fy2 = None, None, None, None
                    if face_xxyy is not None:
                        fx1, fx2, fy1, fy2 = face_xxyy
                        fx1 = max(x1, min(fx1, x2))
                        fx2 = max(x1, min(fx2, x2))
                        fy1 = max(y1, min(fy1, y2))
                        fy2 = max(y1, min(fy2, y2))

                    if not person.verified and current_time - person.last_face_check >= 3 and face_xxyy is not None:
                        face_crop = frame[fy1:fy2, fx1:fx2].copy()
                        if face_crop.size > 0:
                            try:
                                self.face_detector.add_to_queue(track_id, face_crop)
                                person.last_face_check = current_time
                            except queue.Full:
                                pass

                    item = {
                        "id": track_id,
                        "x1": x1,
                        "x2": x2,
                        "y1": y1,
                        "y2": y2,
                        "fx1": fx1,
                        "fx2": fx2,
                        "fy1": fy1,
                        "fy2": fy2
                    }
                    items.append(item)

            payload = {"frame": frame, "items": items}

            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass

            try:
                self.frame_queue.put_nowait(payload)
            except queue.Full:
                pass

            if refresh_appearances:
                self.appearance_frame_count = 0

    def get_frame(self, timeout=1.0):
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        self.running = False

        if self.thread is not None:
            self.thread.join(timeout=2.0)

class Display:
    def __init__(self, detection, people_manager):
        self.detection = detection
        self.people_manager = people_manager

        self.frame_count = 0
        self.fps_start_time = time.time()
        self.fps = 0
        self.running = False

    def start(self):
        self.running = True

    @staticmethod
    def get_color(person):
        if person.verified:
            return 255, 0, 255  # Purple.
        return 0, 255, 0  # Green.

    def display_loop(self):
        while self.running:
            data = self.detection.get_frame()
            if data is None:
                continue

            self.frame_count += 1
            elapsed = time.time() - self.fps_start_time

            if elapsed >= 1.0:
                self.fps = self.frame_count / elapsed
                self.frame_count = 0
                self.fps_start_time = time.time()

            frame = data["frame"]
            items = data["items"]
            for item in items:
                track_id = item["id"]
                x1 = item["x1"]
                y1 = item["y1"]
                x2 = item["x2"]
                y2 = item["y2"]
                fx1 = item["fx1"]
                fx2 = item["fx2"]
                fy1 = item["fy1"]
                fy2 = item["fy2"]

                person = self.people_manager.get_person(track_id)
                color = self.get_color(person)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                if fx1 is not None:
                    cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), color, 2)

                label = f"{person.name} (Person {person.identity_id} - ID {person.track_id})"
                cv2.putText(frame, label, (x1, max(y1 - 10, 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            fps_text = f"FPS: {self.fps:.1f}"

            (text_width, text_height), baseline = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(frame, (10, 10), (20 + text_width, 20 + text_height + baseline), (0, 0, 0), -1)

            cv2.putText(frame, fps_text, (15, 15 + text_height), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            cv2.imshow("Person Recognizer", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.running = False

    def stop(self):
        self.running = False
