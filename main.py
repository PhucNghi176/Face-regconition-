import os
import cv2
import numpy as np
import torch
import PySimpleGUI as sg
from facenet_pytorch import MTCNN, InceptionResnetV1
from datetime import datetime, timezone, timedelta
import requests

# === Cấu hình ===
known_faces_dir = 'known_faces'
embeddings_file = 'known_face_embeddings.npy'
recognition_threshold = 0.6
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
api_endpoint = 'https://localhost:7151/api/v1/Employee/checkin'

# === Khởi tạo mô hình ===
mtcnn = MTCNN(image_size=160, margin=0, keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# === Tải embeddings ===
known_face_embeddings = {}
if os.path.exists(embeddings_file):
    known_face_embeddings = np.load(embeddings_file, allow_pickle=True).item()

# === PySimpleGUI Layout ===
sg.theme('DarkAmber')

layout = [
    [sg.Image(filename='', key='-IMAGE-')],
    [
        sg.Button('Chụp & Thêm Khuôn Mặt', key='-CAPTURE-'),
        sg.Button('Gửi Check-In', key='-SEND-'),
        sg.Button('Thoát', key='-EXIT-')
    ],
    [sg.Text('Tên:', size=(15, 1)), sg.InputText(key='-NAME-')]
]

window = sg.Window('Nhận Diện Khuôn Mặt Check-In', layout)
cap = cv2.VideoCapture(0)
last_sent_name = None

# === Main Event Loop ===
while True:
    event, values = window.read(timeout=20)

    if event in (sg.WIN_CLOSED, '-EXIT-'):
        break

    # === Chụp và Thêm Khuôn Mặt ===
    if event == '-CAPTURE-':
        _, frame = cap.read()
        imgbytes = cv2.imencode('.png', frame)[1].tobytes()
        window['-IMAGE-'].update(data=imgbytes)

        person_name = values['-NAME-']
        boxes, _ = mtcnn.detect(frame)
        if person_name and boxes is not None:
            person_dir = os.path.join(known_faces_dir, person_name)
            os.makedirs(person_dir, exist_ok=True)
            image_name = f"{person_dir}/face_{len(os.listdir(person_dir))}.jpg"
            cv2.imwrite(image_name, frame)

            face_embeddings = resnet(mtcnn(frame, return_prob=False)).detach().cpu().numpy()
            known_face_embeddings[person_name] = face_embeddings[0]
            np.save(embeddings_file, known_face_embeddings)

    # === Gửi Check-In API Request ===
    if event == '-SEND-':
        _, frame = cap.read()
        boxes, _ =mtcnn.detect(frame)
        if boxes is not None:
            face_embeddings = resnet(mtcnn(frame, return_prob=False)).detach().cpu().numpy()

            for i, box in enumerate(boxes):
                face_embedding = face_embeddings[i]
                name = "Unknown"
                min_dist = recognition_threshold

                for (known_name, known_embedding) in known_face_embeddings.items():
                    dist = np.linalg.norm(known_embedding - face_embedding)
                    if dist < min_dist:
                        min_dist = dist
                        name = known_name
                if name != "Unknown":
                    try:
                        ict_now = datetime.now(timezone(timedelta(hours=7)))
                        data = {
                            "employeeID": name,
                            "actual_CheckIn": ict_now.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
                        }
                        headers = {"Content-Type": "application/json"}
                        url = requests.post(api_endpoint, json=data, headers=headers, verify=False)

                        value = url.json().get('value')
                        sg.popup_notify(value,display_duration_in_ms=50)
                    except requests.exceptions.RequestException as e:
                        print(f"Lỗi khi gửi yêu cầu đến API: {e}")

    # === Cập nhật Camera Feed ===
    _, frame = cap.read()
    boxes, _ = mtcnn.detect(frame)
    if boxes is not None:
        face_embeddings = resnet(mtcnn(frame, return_prob=False)).detach().cpu().numpy()

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            face_embedding = face_embeddings[i]

            name = "Unknown"
            min_dist = recognition_threshold

            for (known_name, known_embedding) in known_face_embeddings.items():
                dist = np.linalg.norm(known_embedding - face_embedding)
                if dist < min_dist:
                    min_dist = dist
                    name = known_name
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    imgbytes = cv2.imencode('.png', frame)[1].tobytes()
    window['-IMAGE-'].update(data=imgbytes)

cap.release()
window.close()
