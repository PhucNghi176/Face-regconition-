import os
import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

# ========= Cấu hình =========
known_faces_dir = 'known_faces'
embeddings_file = 'known_face_embeddings.npy'
recognition_threshold = 0.6  # Ngưỡng nhận diện khuôn mặt
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# ========= Khởi tạo mô hình =========
mtcnn = MTCNN(image_size=160, margin=0, keep_all=True, device=device)  # Chuyển MTCNN lên GPU nếu có
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# ========= Tải embeddings =========
known_face_embeddings = {}
if os.path.exists(embeddings_file):
    known_face_embeddings = np.load(embeddings_file, allow_pickle=True).item()

# ========= Khởi tạo camera =========
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Không thể đọc frame. Thoát chương trình.")
        break

    # ========= Phát hiện khuôn mặt =========
    boxes, _ = mtcnn.detect(frame)

    # ========= Xử lý khuôn mặt =========
    if boxes is not None:
        face_embeddings = resnet(mtcnn(frame, return_prob=False)).detach().cpu().numpy()

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            face_embedding = face_embeddings[i]

            # ========= Nhận diện khuôn mặt =========
            name = "Unknown"
            min_dist = recognition_threshold

            for (known_name, known_embedding) in known_face_embeddings.items():
                dist = np.linalg.norm(known_embedding - face_embedding)
                if dist < min_dist:
                    min_dist = dist
                    name = known_name

            # ========= Vẽ và hiển thị kết quả =========
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # ========= Hiển thị frame =========
    cv2.imshow('Face Detection', frame)

    key = cv2.waitKey(1) & 0xFF

    # ========= Chụp ảnh và thêm khuôn mặt mới =========
    if key == ord('c'):
        person_name = input("Nhập tên: ")
        person_dir = os.path.join(known_faces_dir, person_name)
        os.makedirs(person_dir, exist_ok=True)
        image_name = f"{person_dir}/face_{len(os.listdir(person_dir))}.jpg"
        cv2.imwrite(image_name, frame)

        # Thêm embeddings vào danh sách known_faces (chỉ khi phát hiện được mặt)
        if boxes is not None:
            known_face_embeddings[person_name] = face_embeddings[0]
            np.save(embeddings_file, known_face_embeddings)

    # ========= Thoát chương trình =========
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
