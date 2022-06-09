import numpy as np
import math
import time
import sys
import cv2
import argparse
import pkg_resources
from .stabilizer import Stabilizer

try:
    from tflite_runtime.interpreter import Interpreter
except:
    from tensorflow.lite.python.interpreter import Interpreter

fps = ""
detectfps = ""
framecount = 0
detectframecount = 0
time1 = 0
time2 = 0
LABELS = ["face"]


class FaceDetector:
    def __init__(self, model_face_detect, num_threads):
        # Init Face Detector
        self.interpreter_face_detect = Interpreter(model_path=model_face_detect)
        try:
            self.interpreter_face_detect.set_num_threads(num_threads)
        except:
            print(
                "WARNING: The installed PythonAPI of Tensorflow/Tensorflow Lite runtime does not support Multi-Thread processing."
            )
            print("WARNING: It works in single thread mode.")
            print(
                "WARNING: If you want to use Multi-Thread to improve performance on aarch64/armv7l platforms, please refer to one of the below to implement a customized Tensorflow/Tensorflow Lite runtime."
            )
            print("https://github.com/PINTO0309/Tensorflow-bin.git")
            print("https://github.com/PINTO0309/TensorflowLite-bin.git")
            pass
        self.interpreter_face_detect.allocate_tensors()
        self.input_details = self.interpreter_face_detect.get_input_details()[0][
            "index"
        ]
        self.box = self.interpreter_face_detect.get_output_details()[0]["index"]
        self.scores = self.interpreter_face_detect.get_output_details()[2]["index"]
        self.count = self.interpreter_face_detect.get_output_details()[3]["index"]


class HeadPoseEstimator:
    def __init__(self, model_head_pose, num_threads):
        # Init Head Pose Estimator
        self.interpreter_head_pose = Interpreter(model_path=model_head_pose)
        try:
            self.interpreter_head_pose.set_num_threads(num_threads)
        except:
            pass
        self.interpreter_head_pose.allocate_tensors()
        self.input_details = self.interpreter_head_pose.get_input_details()[0]["index"]
        self.predictions = self.interpreter_head_pose.get_output_details()[0]["index"]


def get_square_box(box):
    """Get a square box out of the given box, by expanding it."""
    left_x = box[0]
    top_y = box[1]
    right_x = box[2]
    bottom_y = box[3]

    box_width = right_x - left_x
    box_height = bottom_y - top_y

    # Check if box is already a square. If not, make it a square.
    diff = box_height - box_width
    delta = int(abs(diff) / 2)
    if diff == 0:  # Already a square.
        return box
    elif diff > 0:  # Height > width, a slim box.
        left_x -= delta
        right_x += delta
        if diff % 2 == 1:
            right_x += 1
    else:  # Width > height, a short box.
        top_y -= delta
        bottom_y += delta
        if diff % 2 == 1:
            bottom_y += 1
    return [left_x, top_y, right_x, bottom_y]


def draw_annotation_box(
    image,
    rotation_vector,
    translation_vector,
    camera_matrix,
    dist_coeefs,
    color=(255, 255, 255),
    line_width=2,
):
    """Draw a 3D box as annotation of pose"""
    point_3d = []
    rear_size = 75
    rear_depth = 0
    point_3d.append((-rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, -rear_size, rear_depth))

    front_size = 100
    front_depth = 100
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d.append((-front_size, front_size, front_depth))
    point_3d.append((front_size, front_size, front_depth))
    point_3d.append((front_size, -front_size, front_depth))
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)

    # Map to 2d image points
    (point_2d, _) = cv2.projectPoints(
        point_3d, rotation_vector, translation_vector, camera_matrix, dist_coeefs
    )
    point_2d = np.int32(point_2d.reshape(-1, 2))

    # Draw all the lines
    cv2.polylines(image, [point_2d], True, color, line_width, cv2.LINE_AA)
    cv2.line(
        image, tuple(point_2d[1]), tuple(point_2d[6]), color, line_width, cv2.LINE_AA
    )
    cv2.line(
        image, tuple(point_2d[2]), tuple(point_2d[7]), color, line_width, cv2.LINE_AA
    )
    cv2.line(
        image, tuple(point_2d[3]), tuple(point_2d[8]), color, line_width, cv2.LINE_AA
    )


def head_pose():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_to_file", default="face_photo.jpeg", help="Path to the file"
    )

    args = parser.parse_args()

    path_to_file = args.path_to_file
    model_face_detect = (
        "ssdlite_mobilenet_v2_face_300_integer_quant_with_postprocess.tflite"
    )
    model_head_pose = "head_pose_estimator_integer_quant.tflite"
    image_width = 640
    image_height = 480
    num_threads = 1

    # 3D model points.
    model_points = (
        np.array(
            [
                (0.0, 0.0, 0.0),  # Nose tip
                (0.0, -330.0, -65.0),  # Chin
                (-225.0, 170.0, -135.0),  # Left eye left corner
                (225.0, 170.0, -135.0),  # Right eye right corner
                (-150.0, -150.0, -125.0),  # Mouth left corner
                (150.0, -150.0, -125.0),  # Mouth right corner
            ]
        )
        / 4.5
    )

    raw_value = []
    model_path = pkg_resources.resource_filename("src", "model.txt")
    with open(model_path) as file:
        for line in file:
            raw_value.append(line)
    model_points_68 = np.array(raw_value, dtype=np.float32)
    model_points_68 = np.reshape(model_points_68, (3, -1)).T
    # Transform the model into a front view.
    model_points_68[:, 2] *= -1

    # Camera internals
    focal_length = image_width
    camera_center = (image_width / 2, image_height / 2)
    camera_matrix = np.array(
        [
            [focal_length, 0, camera_center[0]],
            [0, focal_length, camera_center[1]],
            [0, 0, 1],
        ],
        dtype="double",
    )

    # Assuming no lens distortion
    dist_coeefs = np.zeros((4, 1))

    # Rotation vector and translation vector
    r_vec = np.array([[0.01891013], [0.08560084], [-3.14392813]])
    t_vec = np.array([[-14.97821226], [-10.62040383], [-2053.03596872]])

    # Introduce scalar stabilizers for pose.
    pose_stabilizers = [
        Stabilizer(state_num=2, measure_num=1, cov_process=0.1, cov_measure=0.1)
        for _ in range(6)
    ]

    # Init Face Detector
    face_detector = FaceDetector(model_face_detect, num_threads)
    interpreter_face_detect = face_detector.interpreter_face_detect
    face_detector_input_details = face_detector.input_details
    face_detector_box = face_detector.box
    face_detector_scores = face_detector.scores
    face_detector_count = face_detector.count

    # Init Head Pose Estimator
    head_pose_estimator = HeadPoseEstimator(model_head_pose, num_threads)
    interpreter_head_pose = head_pose_estimator.interpreter_head_pose
    head_pose_estimator_input_details = head_pose_estimator.input_details
    head_pose_estimator_predictions = head_pose_estimator.predictions

    start_time = time.perf_counter()

    image = cv2.imread(path_to_file, 0)

    # Resize and normalize image for network input
    frame = cv2.resize(image, (300, 300))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = np.expand_dims(frame, axis=0)
    frame = frame.astype(np.float32)
    cv2.normalize(frame, frame, -1, 1, cv2.NORM_MINMAX)

    # run model - Face Detector
    interpreter_face_detect.set_tensor(face_detector_input_details, frame)
    interpreter_face_detect.invoke()

    # get results - Face Detector
    boxes = interpreter_face_detect.get_tensor(face_detector_box)[0]
    scores = interpreter_face_detect.get_tensor(face_detector_scores)[0]
    count = interpreter_face_detect.get_tensor(face_detector_count)[0]

    for i, (box, score) in enumerate(zip(boxes, scores)):
        probability = score
        if probability >= 0.6:
            if (
                not math.isnan(box[0])
                and not math.isnan(box[1])
                and not math.isnan(box[2])
                and not math.isnan(box[3])
            ):
                pass
            else:
                continue
            ymin = int(box[0] * image_height)
            xmin = int(box[1] * image_width)
            ymax = int(box[2] * image_height)
            xmax = int(box[3] * image_width)
            if ymin > ymax:
                continue
            if xmin > xmax:
                continue

            offset_y = int(abs(ymax - ymin) * 0.1)
            facebox = get_square_box([xmin, ymin + offset_y, xmax, ymax + offset_y])
            if not (
                facebox[0] >= 0
                and facebox[1] >= 0
                and facebox[2] <= image_width
                and facebox[3] <= image_height
            ):
                continue

            face_img = image[facebox[1] : facebox[3], facebox[0] : facebox[2]]
            face_img = cv2.resize(face_img, (128, 128))
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            face_img = np.expand_dims(face_img, axis=0)
            face_img = face_img.astype(np.float32)

            # run model - Head Pose
            interpreter_head_pose.set_tensor(
                head_pose_estimator_input_details, face_img
            )
            interpreter_head_pose.invoke()

            # get results - Head Pose
            predictions = interpreter_head_pose.get_tensor(
                head_pose_estimator_predictions
            )[0]
            marks = np.array(predictions).flatten()[:136]
            marks = np.reshape(marks, (-1, 2))

            # Convert the marks locations from local CNN to global image.
            marks *= facebox[2] - facebox[0]
            marks[:, 0] += facebox[0]
            marks[:, 1] += facebox[1]

            # Try pose estimation with 68 points.
            if r_vec is None:
                (_, rotation_vector, translation_vector) = cv2.solvePnP(
                    model_points_68, marks, camera_matrix, dist_coeefs
                )
                r_vec = rotation_vector
                t_vec = translation_vector
            (_, rotation_vector, translation_vector) = cv2.solvePnP(
                model_points_68,
                marks,
                camera_matrix,
                dist_coeefs,
                rvec=r_vec,
                tvec=t_vec,
                useExtrinsicGuess=True,
            )
            pose = (rotation_vector, translation_vector)

            # Stabilize the pose.
            steady_pose = []
            pose_np = np.array(pose).flatten()
            for value, ps_stb in zip(pose_np, pose_stabilizers):
                ps_stb.update([value])
                steady_pose.append(ps_stb.state[0])
            steady_pose = np.reshape(steady_pose, (-1, 3))

            # Draw boxes
            draw_annotation_box(
                image,
                steady_pose[0],
                steady_pose[1],
                camera_matrix,
                dist_coeefs,
                color=(128, 255, 128),
            )
            cv2.putText(
                image,
                detectfps,
                (image_width - 170, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (38, 0, 255),
                1,
                cv2.LINE_AA,
            )

        if i >= (count - 1):
            break

    return image


if __name__ == "__main__":
    image = head_pose()
    cv2.imshow("Image with head pose", image)
    cv2.waitKey(100000)
