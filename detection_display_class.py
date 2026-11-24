import cv2
from ultralytics import YOLO
from deepface import DeepFace
import time
import threading
import queue
import os
import numpy as np
from types import SimpleNamespace
from ultralytics.trackers.byte_tracker import BYTETracker

class Person:
    def __init__(self, identity_id, camera_amount):
        self.track_ids = [None] * camera_amount
        self.last_face_check = 0
        self.name = "Unknown"
        self.verified = False

        self.identity_id = identity_id
        self.appearance = [None] * camera_amount
        self.last_seen = time.time()

class PeopleManager:
    def __init__(self, camera_amount):
        self.people = {}
        self.track_to_person = {}  # Holds (camera index, track_id): person
        self.next_identity_id = 1
        self.camera_amount = camera_amount

    def get_person(self, camera_index, track_id):
        return self.track_to_person.get((camera_index, track_id))

    def update_person(self, track_id, appearance, current_tracks_ids, camera_index):
        person = self.get_person(camera_index, track_id)

        if person is not None:
            person.last_seen = time.time()
            person.track_ids[camera_index] = track_id

            if appearance is not None:
                person.appearance[camera_index] = appearance

            return person

        best_person = None
        best_distance = float("inf")
        now = time.time()

        if appearance is not None:
            for candidate in list(self.people.values()):
                old_track_id = candidate.track_ids[camera_index]

                if old_track_id in current_tracks_ids:
                    continue

                if now - candidate.last_seen > 30:
                    self.people.pop(candidate.identity_id, None)
                    self.track_to_person.pop((camera_index, old_track_id), None)
                    continue

                for old_appearance in candidate.appearance:
                    if old_appearance is None:
                        continue

                    distance = cv2.compareHist(
                        appearance,
                        old_appearance,
                        cv2.HISTCMP_BHATTACHARYYA
                    )

                    if distance < best_distance:
                        best_distance = distance
                        best_person = candidate
        if best_person is not None:
            print(f"New person - {best_person.identity_id} - {best_distance}")

        if best_person is not None and best_distance < 0.35:
            person = best_person
            old_track_id = person.track_ids[camera_index]
            person.track_ids[camera_index] = track_id
            person.last_seen = now
            person.appearance[camera_index] = appearance
            self.track_to_person[camera_index, track_id] = person
            if old_track_id is not None:
                self.track_to_person.pop((camera_index, old_track_id), None)

            print(f"Track ID: {track_id} - matches: {person.identity_id} - (distance={best_distance:.3f})")
            return person

        identity_id = self.next_identity_id
        self.next_identity_id += 1
        person = Person(identity_id=identity_id, camera_amount=self.camera_amount)
        person.track_ids[camera_index] = track_id
        person.appearance[camera_index] = appearance
        person.last_seen = now

        self.people[identity_id] = person
        self.track_to_person[camera_index, track_id] = person

        return person

    def update_verification(self, track_id, name, verified, camera_index):
        person = self.get_person(camera_index, track_id)

        if person is None:
            return

        person.name = name
        person.verified = verified

        if not verified or not name or name == "Unknown":
            return

        for candidate in list(self.people.values()):
            if candidate is person:
                continue

            if not candidate.verified or candidate.name != name:
                continue

            for i in range(self.camera_amount):
                old_track_id = person.track_ids[i]

                if old_track_id is not None:
                    candidate.track_ids[i] = old_track_id
                    self.track_to_person[(i, old_track_id)] = candidate

                if person.appearance[i] is not None:
                    candidate.appearance[i] = person.appearance[i]

            candidate.last_seen = max(candidate.last_seen, person.last_seen)
            candidate.last_face_check = max(candidate.last_face_check, person.last_face_check)

            self.people.pop(person.identity_id, None)

            print(f"Merged: Person {person.identity_id} - based on face"
                  f"into {candidate.identity_id} "
                  f"(name={name})")

    def link_people(self, track_id1, camera_index1, track_id2, camera_index2):
        person = self.get_person(camera_index1, track_id1)
        person1 = self.get_person(camera_index2, track_id2)

        if person is None or person1 is None:
            return

        if person is person1:
            return

        if (person.verified and person1.verified
                and person.name != person1.name):
            return

        if person1.verified and not person.verified:
            person, person1 = person1, person

        # Transfer camera tracks and appearances
        for i in range(self.camera_amount):
            old_track_id = person1.track_ids[i]

            if old_track_id is not None:
                person.track_ids[i] = old_track_id
                self.track_to_person[(i, old_track_id)] = person

            if person1.appearance[i] is not None:
                person.appearance[i] = person1.appearance[i]

        person.last_seen = max(person.last_seen, person1.last_seen)
        person.last_face_check = max(person.last_face_check, person1.last_face_check)

        self.people.pop(person1.identity_id, None)

        print(
            f"Merged: Person {person1.identity_id} - based on stereo "
            f"into {person.identity_id} "
            f"(name={person.name})")

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

            track_id, camera_index, face_crop = task

            try:
                result = DeepFace.represent(img_path=face_crop, model_name="Facenet", detector_backend="opencv", enforce_detection=True)
                detected_embedding = np.array(result[0]["embedding"], dtype=np.float32)

                name = self.find_best_match(detected_embedding)

                if name is not None:
                    name = name[0].upper() + name[1:].lower()
                    self.people_manager.update_verification(track_id, name, True, camera_index)

            except Exception as error:
                print(f"Face check failed - Camera: {camera_index}, ID: {track_id} -", error)
            
            finally:
                self.face_queue.task_done()

    def add_to_queue(self, track_id, camera_index, face_crop):
        self.face_queue.put_nowait((track_id, camera_index, face_crop))

    def stop(self):
        self.running = False

        try:
            self.face_queue.put_nowait(None)
        except queue.Full:
            pass

        if self.thread is not None:
            self.thread.join(timeout=2.0)

class FrameCapture:
    def __init__(self, camera_index, camera_amount, queue_size, stereo_list=None):
        self.camera_index = camera_index
        self.frame_queue = queue.Queue(maxsize=queue_size)
        self.camera_amount = camera_amount
        self.stereo_list = stereo_list

        self.cameras: list[cv2.VideoCapture | None] = [None] * camera_amount
        self.thread = None
        self.running = False

    def start(self):
        for i, cam_index in enumerate(self.camera_index):
            camera = cv2.VideoCapture(cam_index, cv2.CAP_MSMF)

            if not camera.isOpened():
                for opened_camera in self.cameras:
                    if opened_camera is not None:
                        opened_camera.release()
                raise RuntimeError(f"Could not open camera {cam_index}")

            # Use 1280x720 to improve face detection without significantly affecting performance.
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            camera.set(cv2.CAP_PROP_FPS, 30)
            camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Disable autofocus
            camera.set(cv2.CAP_PROP_FOCUS, 100)
            camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Disable automatic light adjustment
            camera.set(cv2.CAP_PROP_EXPOSURE, -5)  # Set the manual light adjustment to keep the images as similar as possible

            self.cameras[i] = camera

        self.running = True

        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

        return self

    def _capture_loop(self):
        while self.running:
            frames = []
            for i, camera in enumerate(self.cameras):
                if camera is None:
                    print(f"Camera {self.camera_index[i]} is None")
                    continue
                success, frame = camera.read()

                if not success:
                    print(f"Camera {self.camera_index[i]} failed to read a frame")
                    continue
                stereo = None
                if self.stereo_list is not None:
                    for j in range(len(self.stereo_list)):
                        if self.camera_index[i] in self.stereo_list[j]:
                            stereo = self.stereo_list[j]

                frame_data = {
                    "frame": frame,
                    "camera": self.camera_index[i],
                    "stereo": stereo,
                    "timestamp": time.perf_counter()
                }

                frames.append(frame_data)

            if not frames:
                continue

            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass

            try:
                self.frame_queue.put_nowait(frames)
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

        for i, camera in enumerate(self.cameras):
            if camera is not None:
                camera.release()
                self.cameras[i] = None

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
        self.trackers = []

        self.running = False
        self.thread = None

    def start(self):
        self.model = YOLO("yolo11m-pose.engine")

        tracker_args = SimpleNamespace(tracker_type="bytetrack", track_high_thresh=0.25, track_low_thresh=0.1, new_track_thresh=0.25, track_buffer=30, match_thresh=0.8, fuse_score=True)
        self.trackers = [BYTETracker(args=tracker_args) for _ in range(self.capture.camera_amount)]

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

            if len(data) != len(self.trackers):
                continue

            batch = [frame_data["frame"] for frame_data in data]
            self.appearance_frame_count += 1
            refresh_appearances = (
                self.appearance_frame_count >= self.appearance_refresh_interval
            )

            payload = []
            current_time = time.time()
            results = self.model.predict(batch, classes=[0], verbose=False, conf=self.conf, iou=self.iou)

            for i, result in enumerate(results):
                frame_data = data[i]
                frame = frame_data["frame"]
                camera_index = frame_data["camera"]
                camera = {"capture": frame_data}
                items = []
                position_points = []

                frame_height, frame_width = frame.shape[:2]
                tracker = self.trackers[camera_index]

                detections = result.boxes.cpu().numpy()
                tracks = tracker.update(detections, frame)
                current_track_ids = {int(track[4]) for track in tracks}
                keypoints = None
                face_keypoints = None
                if result.keypoints is not None:
                    keypoints = result.keypoints.xy[:, [0, 1, 2, 3, 4, 5, 6, 11, 12]].cpu().numpy()
                    face_keypoints = keypoints[:, :5].copy()

                for track in tracks:
                    x1, y1, x2, y2 = map(int, track[:4])
                    track_id = int(track[4])
                    detections_index = int(track[7])

                    x1 = max(0, min(x1, frame_width))
                    x2 = max(0, min(x2, frame_width))
                    y1 = max(0, min(y1, frame_height))
                    y2 = max(0, min(y2, frame_height))

                    if x2 <= x1 or y2 <= y1:
                        continue

                    existing_person = self.people_manager.get_person(camera_index, track_id)
                    should_refresh_appearance = existing_person is None or refresh_appearances
                    appearance = (
                        self.get_appearance(frame, x1, y1, x2, y2) if should_refresh_appearance
                        else None
                    )

                    person = self.people_manager.update_person(track_id, appearance, current_track_ids, camera_index)

                    face_xxyy = None
                    if keypoints is not None and 0 <= detections_index < len(face_keypoints):
                        face_xxyy = self.get_face_crop_xxyy(frame, face_keypoints[detections_index])

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
                                self.face_detector.add_to_queue(track_id, camera_index, face_crop)
                                person.last_face_check = current_time
                            except queue.Full:
                                pass
                    
                    left_shoulder, right_shoulder, left_hip, right_hip = None, None, None, None
                    if keypoints is not None and 0 <= detections_index < len(keypoints):
                        left_shoulder = keypoints[detections_index][5]
                        right_shoulder = keypoints[detections_index][6]
                        left_hip = keypoints[detections_index][7]
                        right_hip = keypoints[detections_index][8]

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
                    points = {
                        "left_shoulder": left_shoulder,
                        "right_shoulder": right_shoulder,
                        "left_hip": left_hip,
                        "right_hip": right_hip
                    }
                    items.append(item)
                    position_points.append(points)
                camera["items"] = items
                camera["position_points"] = position_points
                payload.append(camera)

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

# It works for multiple cameras if they have the same calibration data.
# Will need to be modified if each camera has different calibration data.
class StereoVision:
    def __init__(self, detection, people_manager, queue_size, calibration_data):
        self.detection = detection
        self.people_manager = people_manager
        self.frame_queue = queue.Queue(maxsize=queue_size)

        self.calibration_data = calibration_data
        self.K1 = None
        self.D1 = None
        self.K2 = None
        self.D2 = None
        self.F = None

        self.running = False
        self.thread = None

    def start(self):
        data = np.load(self.calibration_data)

        self.K1 = data["left_matrix"]
        self.D1 = data["left_distortion"]
        self.K2 = data["right_matrix"]
        self.D2 = data["right_distortion"]
        self.F = data["F"]

        self.running = True
        self.thread = threading.Thread(target=self._stereo_loop, daemon=True)
        self.thread.start()

        return self

    def get_epipolar_line(self, x, y, side):  # 0 == left, 1 == right
        point = np.array([x, y, 1.0])

        if side == 0:
            line = self.F @ point
        else:
            line = self.F.T @ point

        return line  # Gives the values a, b, c used in distance

    @staticmethod
    def distance(a, b, c, x2, y2):
        return abs(a*x2 + b*y2 + c) / np.sqrt(a*a + b*b)

    def undistort_point(self, point, camera_index, left_camera_index):
        if point is None:
            return None

        x, y = point[0], point[1]
        if not np.isfinite(x) or not np.isfinite(y) or x <= 0 or y <= 0:
            return None

        if camera_index == left_camera_index:
            matrix = self.K1
            distortion = self.D1
        else:
            matrix = self.K2
            distortion = self.D2

        distorted_point = np.array([[[x, y]]], dtype=np.float32)
        undistorted_point = cv2.undistortPoints(
            distorted_point,
            matrix,
            distortion,
            P=matrix
        )

        return undistorted_point.reshape(2)

    def _stereo_loop(self):
        while self.running:
            data = self.detection.get_frame()
            if data is None:
                continue

            stereo_pair = {}

            for camera in data:
                capture = camera["capture"]
                position_points = camera["position_points"]
                items = camera["items"]
                if capture["stereo"] is None:
                    continue
                pair_key = tuple(capture["stereo"])
                # Save the first camera in the pair.
                if stereo_pair.get(pair_key) is None:
                    info = {
                        "camera_index": capture["camera"],
                        "position_points": position_points,
                        "items": items,
                        "stereo": capture["stereo"]
                    }
                    stereo_pair[pair_key] = info
                    continue
                
                # If the second camera in the pair is found, calculate the match.
                right_camera_index = capture["stereo"][1]
                left_camera_index = capture["stereo"][0]
                lines_list = []

                if capture["camera"] == left_camera_index:
                    left_position_points = position_points
                    left_items = items
                    info = stereo_pair[pair_key]
                    right_position_points = info["position_points"]
                    right_items = info["items"]
                else:
                    right_position_points = position_points
                    right_items = items
                    info = stereo_pair[pair_key]
                    left_position_points = info["position_points"]
                    left_items = info["items"]

                # Calculate epipolar lines for each detected person in the left camera.
                for item, points in zip(left_items, left_position_points):
                    track_id = item["id"]
                    lines = {}
                    for point_name, point in points.items():
                        undistorted_point = self.undistort_point(
                            point,
                            left_camera_index,
                            left_camera_index
                        )
                        if undistorted_point is None:
                            continue

                        line = self.get_epipolar_line(
                            undistorted_point[0],
                            undistorted_point[1],
                            0
                        )
                        lines[point_name] = line

                    obj = {
                        "id": track_id,
                        "camera": left_camera_index,
                        "lines": lines
                        }
                    lines_list.append(obj)
                
                # Match right-camera detections with left-camera epipolar lines.
                distance_list = []
                for item, points in zip(right_items, right_position_points):
                    track_id = item["id"]
                    for element in lines_list:
                        other_camera = element["camera"]
                        other_id = element["id"]
                        lines = element["lines"]
                        distances = []
                        for point_name, line in lines.items():
                            point = points.get(point_name)
                            undistorted_point = self.undistort_point(
                                point,
                                right_camera_index,
                                left_camera_index
                            )
                            if undistorted_point is None:
                                continue

                            x = undistorted_point[0]
                            y = undistorted_point[1]
                            a, b, c = line
                            distance = self.distance(a, b, c, x, y)
                            if np.isfinite(distance):
                                distances.append(distance)

                        if not distances:
                            continue

                        obj = {
                            "camera1": other_camera,
                            "id1": other_id,
                            "camera2": right_camera_index,
                            "id2": track_id,
                            "distances": distances
                        }
                        distance_list.append(obj)

                # Match detections based on the average distance to epipolar lines.
                threshold = 18.0  # pixels

                for match in distance_list:
                    match["score"] = np.mean(match["distances"])

                distance_list.sort(key=lambda candidate: candidate["score"])

                matches = []
                used_left = set()
                used_right = set()
                for match in distance_list:
                    print(match["score"])
                    if match["score"] > threshold:
                        continue

                    if match["id1"] in used_left or match["id2"] in used_right:
                        continue

                    matches.append(match)

                    used_left.add(match["id1"])
                    used_right.add(match["id2"])

                for match in matches:
                    # Link the detections so the program knows they are the same person.
                    self.people_manager.link_people(match["id1"], match["camera1"], match["id2"], match["camera2"])

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

class Display:
    def __init__(self, last_frame_owner, camera_amount, people_manager=None):
        self.last_frame_owner = last_frame_owner
        self.people_manager = people_manager
        self.camera_amount = camera_amount

        self.frame_objs = []
        self.running = False

    def start(self):
        self.fill_frame_obs()
        self.running = True

    def fill_frame_obs(self):
        self.frame_objs.clear()
        for i in range(self.camera_amount):
            obj = {
                "frame_count": 0,
                "fps_start_time": time.time(),
                "fps": 0
                }
            self.frame_objs.append(obj)

    @staticmethod
    def get_color(person):
        if person.verified:
            return 255, 0, 255  # Purple.
        return 0, 255, 0  # Green.

    def display_loop(self):
        while self.running:
            data = self.last_frame_owner.get_frame()
            if data is None:
                continue

            frame_list = []

            for i, camera in enumerate(data):
                self.frame_objs[i]["frame_count"] += 1
                elapsed = time.time() - self.frame_objs[i]["fps_start_time"]

                if elapsed >= 1.0:
                    self.frame_objs[i]["fps"] = self.frame_objs[i]["frame_count"] / elapsed
                    self.frame_objs[i]["frame_count"] = 0
                    self.frame_objs[i]["fps_start_time"] = time.time()

                # Support both raw frames and YOLO-processed frames.
                if "capture" in camera:
                    # Frame from Detection
                    capture = camera["capture"]
                    items = camera.get("items")
                else:
                    capture = camera
                    items = None

                frame = capture["frame"].copy()
                camera_index = capture["camera"]

                if items is not None and self.people_manager is not None:
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

                        person = self.people_manager.get_person(camera_index, track_id)
                        color = self.get_color(person)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        if fx1 is not None:
                            cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), color, 2)

                        label = f"{person.name} (Person {person.identity_id} - ID {person.track_ids[i]})"
                        cv2.putText(frame, label, (x1, max(y1 - 10, 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                frame = cv2.resize(frame, (640, 360))

                fps_text = f"{camera_index} - FPS: {self.frame_objs[i]['fps']:.1f}"

                (text_width, text_height), baseline = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.rectangle(frame, (10, 10), (20 + text_width, 20 + text_height + baseline), (0, 0, 0), -1)

                cv2.putText(frame, fps_text, (15, 15 + text_height), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                frame_list.append(frame)

            combined = cv2.hconcat(frame_list)
            cv2.imshow("Person Recognizer", combined)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.running = False

    def stop(self):
        self.running = False
