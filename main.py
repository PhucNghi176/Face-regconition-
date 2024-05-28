from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch
import cv2
import os
import numpy as np

# Đường dẫn đến thư mục chứa khuôn mặt đã biết
known_faces_dir = 'known_faces'
os.makedirs(known_faces_dir, exist_ok=True)

# Khởi tạo mô hình MTCNN và InceptionResnetV1
mtcnn = MTCNN(image_size=160, margin=0, keep_all=True)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Tạo embedding cho các khuôn mặt đã biết (chỉ khi thư mục không trống)
known_face_embeddings = {}
if os.listdir(known_faces_dir):  # Kiểm tra nếu thư mục có chứa tệp
    for filename in os.listdir(known_faces_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            path = os.path.join(known_faces_dir, filename)
            img = Image.open(path)
            img_cropped = mtcnn(img)
            if img_cropped is not None:
                with torch.no_grad():
                    embedding = resnet(img_cropped.unsqueeze(0))
                known_face_embeddings[os.path.splitext(filename)[0]] = embedding.detach().cpu().numpy()[0]

# Khởi tạo camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Không thể đọc frame từ camera. Thoát chương trình.")
        break

    # Phát hiện khuôn mặt
    boxes, _ = mtcnn.detect(frame)

    # Xử lý từng khuôn mặt (chỉ khi có khuôn mặt được phát hiện)
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            face = frame[y1:y2, x1:x2]
            img_cropped = mtcnn(face)

            if img_cropped is not None:
                img_cropped = img_cropped.squeeze(0)  # Loại bỏ chiều nếu cần thiết

                with torch.no_grad():
                    face_embedding = resnet(img_cropped.unsqueeze(0)).detach().cpu().numpy()[0]

                name = "Unknown"  # Mặc định là "Unknown" nếu không có khuôn mặt nào được biết đến

                if known_face_embeddings:  # Kiểm tra nếu known_face_embeddings không trống
                    min_dist = float('inf')

                    for (name, embedding) in known_face_embeddings.items():
                        dist = np.linalg.norm(embedding - face_embedding)
                        if dist < min_dist:
                            min_dist = dist
                            best_match = name

                    if min_dist < 0.6:  # Ngưỡng nhận diện (có thể điều chỉnh)
                        name = best_match

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Hiển thị frame
    cv2.imshow('Face Detection', frame)

    # Nhấn phím 'c' để chụp ảnh và lưu vào 'known_faces'
    if cv2.waitKey(1) & 0xFF == ord('c'):
        person_name = input("Nhập tên người trong ảnh: ")
        person_dir = os.path.join(known_faces_dir, person_name)
        os.makedirs(person_dir, exist_ok=True)

        image_name = f"{person_dir}/face_{len(os.listdir(person_dir))}.jpg"
        cv2.imwrite(image_name, frame)
        print(f"Ảnh đã được chụp và lưu tại: {image_name}")

        # Thêm khuôn mặt mới vào danh sách known_faces
        img = Image.open(image_name)
        img_cropped = mtcnn(img)
        if img_cropped is not None:
            img_cropped = img_cropped.squeeze(0)
            with torch.no_grad():
                embedding = resnet(img_cropped.unsqueeze(0))
            known_face_embeddings[person_name] = embedding.detach().cpu().numpy()[0]

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
