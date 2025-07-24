import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm
import os

# Senin başlıkların (tab ile ayrılmış)
header = """indexPIP_left_X	leftElbow_Y	ringPIP_left_Y	thumbMP_left_X	ringDIP_right_X	leftEye_Y	indexPIP_right_Y	littleMCP_right_X	middleDIP_left_X	ringPIP_left_X	littleMCP_right_Y	ringTip_left_X	rightEye_X	video_size_height	wrist_right_Y	thumbMP_right_Y	littleTip_right_Y	thumbTip_left_X	middleMCP_right_Y	rightWrist_Y	middleTip_right_Y	root_X	ringMCP_left_Y	indexPIP_left_Y	thumbCMC_right_X	thumbCMC_left_Y	indexDIP_right_Y	neck_Y	ringTip_right_X	middlePIP_right_Y	indexTip_right_X	rightShoulder_Y	littleMCP_left_X	ringDIP_right_Y	thumbCMC_right_Y	indexMCP_left_X	indexDIP_left_Y	nose_Y	ringMCP_right_X	thumbIP_left_X	leftElbow_X	indexDIP_right_X	indexDIP_left_X	video_size_width	middleDIP_right_X	thumbMP_right_X	wrist_left_X	indexMCP_right_X	rightEar_X	leftEar_X	thumbIP_right_Y	rightEye_Y	leftShoulder_Y	ringPIP_right_X	leftWrist_X	middleTip_left_Y	littlePIP_right_X	indexMCP_right_Y	indexTip_left_Y	ringMCP_right_Y	middleMCP_left_X	thumbTip_left_Y	wrist_left_Y	labels	middleMCP_right_X	middleMCP_left_Y	rightWrist_X	littleDIP_right_Y	ringTip_right_Y	leftEar_Y	rightShoulder_X	littlePIP_right_Y	littleTip_right_X	middlePIP_left_Y	indexPIP_right_X	middlePIP_left_X	littleDIP_right_X	middleDIP_left_Y	leftWrist_Y	middleDIP_right_Y	video_fps	littleDIP_left_X	littleMCP_left_Y	ringDIP_left_X	leftEye_X	littleTip_left_Y	thumbMP_left_Y	indexMCP_left_Y	indexTip_right_Y	thumbIP_left_Y	ringTip_left_Y	wrist_right_X	thumbCMC_left_X	rightEar_Y	indexTip_left_X	neck_X	middleTip_left_X	ringDIP_left_Y	middlePIP_right_X	root_Y	rightElbow_X	thumbTip_right_X	littleTip_left_X	littlePIP_left_X	littlePIP_left_Y	leftShoulder_X	middleTip_right_X	thumbTip_right_Y	thumbIP_right_X	nose_X	rightElbow_Y	littleDIP_left_Y	ringPIP_right_Y	ringMCP_left_X"""

columns = header.split('\t')

# MediaPipe init
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

# Landmark mapping tabloları (örnek)
pose_map = {
    "leftElbow": mp_pose.PoseLandmark.LEFT_ELBOW,
    "rightElbow": mp_pose.PoseLandmark.RIGHT_ELBOW,
    "leftShoulder": mp_pose.PoseLandmark.LEFT_SHOULDER,
    "rightShoulder": mp_pose.PoseLandmark.RIGHT_SHOULDER,
    "leftWrist": mp_pose.PoseLandmark.LEFT_WRIST,
    "rightWrist": mp_pose.PoseLandmark.RIGHT_WRIST,
    "leftEar": mp_pose.PoseLandmark.LEFT_EAR,
    "rightEar": mp_pose.PoseLandmark.RIGHT_EAR,
    "leftEye": mp_pose.PoseLandmark.LEFT_EYE,
    "rightEye": mp_pose.PoseLandmark.RIGHT_EYE,
    "nose": mp_pose.PoseLandmark.NOSE,
    # "neck" ve "root" MediaPipe'ta direkt yok, kendin hesaplayabilirsin!
}

hand_map = {
    "wrist": mp_hands.HandLandmark.WRIST,
    "thumbCMC": mp_hands.HandLandmark.THUMB_CMC,
    "thumbMP": mp_hands.HandLandmark.THUMB_MCP,
    "thumbIP": mp_hands.HandLandmark.THUMB_IP,
    "thumbTip": mp_hands.HandLandmark.THUMB_TIP,
    "indexMCP": mp_hands.HandLandmark.INDEX_FINGER_MCP,
    "indexPIP": mp_hands.HandLandmark.INDEX_FINGER_PIP,
    "indexDIP": mp_hands.HandLandmark.INDEX_FINGER_DIP,
    "indexTip": mp_hands.HandLandmark.INDEX_FINGER_TIP,
    "middleMCP": mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
    "middlePIP": mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
    "middleDIP": mp_hands.HandLandmark.MIDDLE_FINGER_DIP,
    "middleTip": mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
    "ringMCP": mp_hands.HandLandmark.RING_FINGER_MCP,
    "ringPIP": mp_hands.HandLandmark.RING_FINGER_PIP,
    "ringDIP": mp_hands.HandLandmark.RING_FINGER_DIP,
    "ringTip": mp_hands.HandLandmark.RING_FINGER_TIP,
    "littleMCP": mp_hands.HandLandmark.PINKY_MCP,
    "littlePIP": mp_hands.HandLandmark.PINKY_PIP,
    "littleDIP": mp_hands.HandLandmark.PINKY_DIP,
    "littleTip": mp_hands.HandLandmark.PINKY_TIP
}

def get_landmark_value(name, suffix, pose_landmarks, hands_landmarks, handedness, frame, frame_width, frame_height , labels):
    # video/frame bilgileri
    if name == "video_size_width":
        return frame_width
    if name == "video_size_height":
        return frame_height
    if name == "video_fps":
        return 25  # istersen video fps sini ekleyebilirsin
    if name == "labels":
        return labels # manuel eklenir
    # neck ve root özel!
    if name == "neck":
        # Neck = iki omuzun ortası
        if pose_landmarks:
            left = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            if suffix == "X":
                return (left.x + right.x) / 2
            else:
                return (left.y + right.y) / 2
        else:
            return 0
    if name == "root":
        # Root = iki kalçanın ortası
        if pose_landmarks:
            left = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
            right = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
            if suffix == "X":
                return (left.x + right.x) / 2
            else:
                return (left.y + right.y) / 2
        else:
            return 0

    # Pose landmarks
    if name in pose_map:
        idx = pose_map[name]
        if pose_landmarks:
            if suffix == "X":
                return pose_landmarks.landmark[idx].x
            else:
                return pose_landmarks.landmark[idx].y
        else:
            return 0

    # El landmarks
    # item formatı ör: indexTip_left_X, thumbTip_right_Y ...
    if "_" in name:
        # ör: indexTip_left_X  → "indexTip", "left"
        key, hand_side = name.split("_")
        idx = hand_map.get(key, None)
        if idx is None:
            return 0
        # Hangi el?
        side_found = False
        for i, h in enumerate(hands_landmarks):
            label = handedness[i].classification[0].label.lower()  # 'left' ya da 'right'
            if label == hand_side:
                side_found = True
                if suffix == "X":
                    return h.landmark[idx].x
                else:
                    return h.landmark[idx].y
        if not side_found:
            return 0
    return 0

def extract_frame_features(frame, pose_landmarks, hands_landmarks, handedness, frame_width, frame_height, labels):
    features = []
    for col in columns:
        # Örnek: "indexPIP_left_X"
        if col.endswith("_X"):
            name = col[:-2]
            val = get_landmark_value(name, "X", pose_landmarks, hands_landmarks, handedness, frame, frame_width, frame_height, labels)
            features.append(val)
        elif col.endswith("_Y"):
            name = col[:-2]
            val = get_landmark_value(name, "Y", pose_landmarks, hands_landmarks, handedness, frame, frame_width, frame_height, labels)
            features.append(val)
        else:
            val = get_landmark_value(col, "", pose_landmarks, hands_landmarks, handedness, frame, frame_width, frame_height, labels)
            features.append(val)
    return features

#wlasl dataset
src_path = '/media/zeynep/External/Zeynep/PHD/Repos/VideoMAEv2/data/wlasl100_64x64_640x480_PIL/train'
dst_path = './data/wlasl100'
data_txt  = '/media/zeynep/External/Zeynep/PHD/Repos/VideoMAEv2/data/wlasl100_64x64_640x480_PIL/train.txt'
dataset="wlasl100"
split = "train"

with open(data_txt, 'r') as file:
    lines = file.readlines()
    count_f = 0
    for row in lines:
        folder_path = os.path.join(src_path , row.split(" ")[0].split("/")[-1])
        # List all files in the folder
        files = sorted(os.listdir(folder_path))

        label = row.strip().split(" ")[-1]

        # Filter out directories if needed
        files = [os.path.join(folder_path, f) for f in files if os.path.isfile(os.path.join(folder_path, f))]

        frame_count = len(files)

        all_features = []
        with mp_pose.Pose(static_image_mode=False) as pose_model, \
             mp_hands.Hands(static_image_mode=False, max_num_hands=2) as hands_model:

            pbar = tqdm(total=frame_count)
            for frame_path in files:
                # read image frame
                with open(frame_path, 'rb') as f:
                    img_bytes = f.read()

                img_np = np.frombuffer(img_bytes, np.uint8)
                image = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB, image)

                frame_height, frame_width, channels = image_rgb.shape

                pose_results = pose_model.process(image_rgb)
                hands_results = hands_model.process(image_rgb)
                pose_landmarks = pose_results.pose_landmarks
                hands_landmarks = hands_results.multi_hand_landmarks if hands_results.multi_hand_landmarks else []
                handedness = hands_results.multi_handedness if hands_results.multi_handedness else []

                features = extract_frame_features(
                    image_rgb, pose_landmarks, hands_landmarks, handedness, frame_width, frame_height, label
                )
                all_features.append(features)
                pbar.update(1)
            pbar.close()
        print("all_features:", all_features)


# all_features artık [frame_count, num_features] şeklinde bir numpy array olarak kullanılabilir
print(np.array(all_features).shape)
# Dilersen kaydedebilirsin:
# import pandas as pd; pd.DataFrame(all_features, columns=columns).to_csv("output.csv", index=False)
