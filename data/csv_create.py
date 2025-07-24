import argparse
import os
import json
from typing import Dict, List
import cv2
import pandas as pd
from tqdm import tqdm
import mediapipe as mp

from normalization.body_normalization import BODY_IDENTIFIERS
from normalization.hand_normalization import HAND_IDENTIFIERS

# Column order expected in the resulting CSV file
HEADERS = [
    'indexPIP_left_X', 'leftElbow_Y', 'ringPIP_left_Y', 'thumbMP_left_X',
    'ringDIP_right_X', 'leftEye_Y', 'indexPIP_right_Y', 'littleMCP_right_X',
    'middleDIP_left_X', 'ringPIP_left_X', 'littleMCP_right_Y', 'ringTip_left_X',
    'rightEye_X', 'video_size_height', 'wrist_right_Y', 'thumbMP_right_Y',
    'littleTip_right_Y', 'thumbTip_left_X', 'middleMCP_right_Y', 'rightWrist_Y',
    'middleTip_right_Y', 'root_X', 'ringMCP_left_Y', 'indexPIP_left_Y',
    'thumbCMC_right_X', 'thumbCMC_left_Y', 'indexDIP_right_Y', 'neck_Y',
    'ringTip_right_X', 'middlePIP_right_Y', 'indexTip_right_X',
    'rightShoulder_Y', 'littleMCP_left_X', 'ringDIP_right_Y',
    'thumbCMC_right_Y', 'indexMCP_left_X', 'indexDIP_left_Y', 'nose_Y',
    'ringMCP_right_X', 'thumbIP_left_X', 'leftElbow_X', 'indexDIP_right_X',
    'indexDIP_left_X', 'video_size_width', 'middleDIP_right_X',
    'thumbMP_right_X', 'wrist_left_X', 'indexMCP_right_X', 'rightEar_X',
    'leftEar_X', 'thumbIP_right_Y', 'rightEye_Y', 'leftShoulder_Y',
    'ringPIP_right_X', 'leftWrist_X', 'middleTip_left_Y', 'littlePIP_right_X',
    'indexMCP_right_Y', 'indexTip_left_Y', 'ringMCP_right_Y',
    'middleMCP_left_X', 'thumbTip_left_Y', 'wrist_left_Y', 'labels',
    'middleMCP_right_X', 'middleMCP_left_Y', 'rightWrist_X',
    'littleDIP_right_Y', 'ringTip_right_Y', 'leftEar_Y', 'rightShoulder_X',
    'littlePIP_right_Y', 'littleTip_right_X', 'middlePIP_left_Y',
    'indexPIP_right_X', 'middlePIP_left_X', 'littleDIP_right_X',
    'middleDIP_left_Y', 'leftWrist_Y', 'middleDIP_right_Y', 'video_fps',
    'littleDIP_left_X', 'littleMCP_left_Y', 'ringDIP_left_X', 'leftEye_X',
    'littleTip_left_Y', 'thumbMP_left_Y', 'indexMCP_left_Y', 'indexTip_right_Y',
    'thumbIP_left_Y', 'ringTip_left_Y', 'wrist_right_X', 'thumbCMC_left_X',
    'rightEar_Y', 'indexTip_left_X', 'neck_X', 'middleTip_left_X',
    'ringDIP_left_Y', 'middlePIP_right_X', 'root_Y', 'rightElbow_X',
    'thumbTip_right_X', 'littleTip_left_X', 'littlePIP_left_X',
    'littlePIP_left_Y', 'leftShoulder_X', 'middleTip_right_X',
    'thumbTip_right_Y', 'thumbIP_right_X', 'nose_X', 'rightElbow_Y',
    'littleDIP_left_Y', 'ringPIP_right_Y', 'ringMCP_left_X'
]

POSE_MAP = {
    'nose': mp.solutions.holistic.PoseLandmark.NOSE,
    'rightEye': mp.solutions.holistic.PoseLandmark.RIGHT_EYE,
    'leftEye': mp.solutions.holistic.PoseLandmark.LEFT_EYE,
    'rightEar': mp.solutions.holistic.PoseLandmark.RIGHT_EAR,
    'leftEar': mp.solutions.holistic.PoseLandmark.LEFT_EAR,
    'rightShoulder': mp.solutions.holistic.PoseLandmark.RIGHT_SHOULDER,
    'leftShoulder': mp.solutions.holistic.PoseLandmark.LEFT_SHOULDER,
    'rightElbow': mp.solutions.holistic.PoseLandmark.RIGHT_ELBOW,
    'leftElbow': mp.solutions.holistic.PoseLandmark.LEFT_ELBOW,
    'rightWrist': mp.solutions.holistic.PoseLandmark.RIGHT_WRIST,
    'leftWrist': mp.solutions.holistic.PoseLandmark.LEFT_WRIST,
}

HAND_MAP = {
    'wrist': 0,
    'indexTip': 8,
    'indexDIP': 7,
    'indexPIP': 6,
    'indexMCP': 5,
    'middleTip': 12,
    'middleDIP': 11,
    'middlePIP': 10,
    'middleMCP': 9,
    'ringTip': 16,
    'ringDIP': 15,
    'ringPIP': 14,
    'ringMCP': 13,
    'littleTip': 20,
    'littleDIP': 19,
    'littlePIP': 18,
    'littleMCP': 17,
    'thumbTip': 4,
    'thumbIP': 3,
    'thumbMP': 2,
    'thumbCMC': 1,
}

def init_sequence() -> Dict[str, List[float]]:
    seq = {}
    for name in BODY_IDENTIFIERS:
        seq[f'{name}_X'] = []
        seq[f'{name}_Y'] = []
    for name in HAND_IDENTIFIERS:
        seq[f'{name}_left_X'] = []
        seq[f'{name}_left_Y'] = []
        seq[f'{name}_right_X'] = []
        seq[f'{name}_right_Y'] = []
    seq['root_X'] = []
    seq['root_Y'] = []
    return seq

def append_values(seq: Dict[str, List[float]], data: Dict[str, float]):
    for k in seq.keys():
        seq[k].append(data.get(k, 0.0))

def extract_landmarks(results, width, height):
    output = {}
    pose = results.pose_landmarks.landmark if results.pose_landmarks else None
    if pose:
        left_sh = pose[mp.solutions.holistic.PoseLandmark.LEFT_SHOULDER]
        right_sh = pose[mp.solutions.holistic.PoseLandmark.RIGHT_SHOULDER]
        left_hip = pose[mp.solutions.holistic.PoseLandmark.LEFT_HIP]
        right_hip = pose[mp.solutions.holistic.PoseLandmark.RIGHT_HIP]
        neck_x = (left_sh.x + right_sh.x) / 2.0
        neck_y = (left_sh.y + right_sh.y) / 2.0
        root_x = (left_hip.x + right_hip.x) / 2.0
        root_y = (left_hip.y + right_hip.y) / 2.0
    else:
        neck_x = neck_y = None
        neck_x = neck_y = root_x = root_y = None
    for name, idx in POSE_MAP.items():
        if pose:
            lm = pose[idx]
            output[f'{name}_X'] = lm.x
            output[f'{name}_Y'] = lm.y
        else:
            output[f'{name}_X'] = 0.0
            output[f'{name}_Y'] = 0.0
    output['neck_X'] = neck_x if neck_x is not None else 0.0
    output['neck_Y'] = neck_y if neck_y is not None else 0.0
    output['root_X'] = root_x if root_x is not None else 0.0
    output['root_Y'] = root_y if root_y is not None else 0.0

    for name, idx in HAND_MAP.items():
        if results.left_hand_landmarks:
            lm = results.left_hand_landmarks.landmark[idx]
            output[f'{name}_left_X'] = lm.x
            output[f'{name}_left_Y'] = lm.y
        else:
            output[f'{name}_left_X'] = 0.0
            output[f'{name}_left_Y'] = 0.0
        if results.right_hand_landmarks:
            lm = results.right_hand_landmarks.landmark[idx]
            output[f'{name}_right_X'] = lm.x
            output[f'{name}_right_Y'] = lm.y
        else:
            output[f'{name}_right_X'] = 0.0
            output[f'{name}_right_Y'] = 0.0
    return output

def process_video(path: str):
    cap = cv2.VideoCapture(path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    sequence = init_sequence()

def main():
    wlasl_root ="/media/zeynep/External/Zeynep/PHD/Repos/VideoMAEv2/data/wlasl100_64x64_640x480_PIL"
    split="train"
    output_csv ="./WLASL100_train.csv"
    specs_path = os.path.join(wlasl_root, split+".txt")
    with open(specs_path, 'r') as f:
        data = json.load(f)

    rows = []
    for item in tqdm(data):
        label = item.get('gloss', item.get('label', ''))
        for inst in item['instances']:
            if inst['split'] !=split:
                continue
            video_path = os.path.join(wlasl_root, 'videos', f"{inst['video_id']}.mp4")
            if not os.path.exists(video_path):
                continue
            seq, w, h, fps = process_video(video_path)
            row = {
                'labels': label,
                'video_size_width': w,
                'video_size_height': h,
                'video_fps': fps,
            }
            row.update({k: v for k, v in seq.items()})
            rows.append(row)

    df = pd.DataFrame(rows)
    df = df.reindex(columns=HEADERS)
    df.to_csv(output_csv, index=False, encoding='utf-8')

if __name__ == '__main__':
    main()
