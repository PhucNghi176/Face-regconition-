# Face Recognition Check-In System

This project implements a real-time face recognition system for employee check-ins, integrated with a user-friendly PySimpleGUI interface. The system utilizes deep learning models to detect, recognize faces, and send check-in data to a specified API endpoint.

## Features

* **Real-time Face Detection & Recognition:**  Accurately detects and recognizes faces in a live camera feed using the MTCNN and FaceNet models.
* **Check-In API Integration:**  Sends check-in data (employee ID, timestamp) to a designated API endpoint after successful face recognition.
* **PySimpleGUI Interface:**  Provides an intuitive graphical user interface for:
    - Adding new faces to the system's database.
    - Triggering check-in requests manually.
    - Visualizing the camera feed with face detection/recognition results.
* **Customizable:** Easily adapt the API endpoint, recognition threshold, and other parameters to fit your specific environment.

## Prerequisites

* **Python 3.x:** Make sure you have Python installed.
* **Libraries:** Install the required Python libraries:

   ```bash
   pip install PySimpleGUI numpy opencv-python torch torchvision facenet-pytorch requests
Optionally, install PyTorch with CUDA support if you have a compatible NVIDIA GPU.
## Usage
1.Clone the Repository:
```Bash
git clone https://github.com/PhucNghi176/Face-regconition-.git 
cd Face-regconition-
```
2.Prepare Images:
<li> Create a folder named known_faces in the project directory.
<li> Place images of known employees within subfolders named after their respective IDs (e.g., known_faces/employee123/).</li><br>
3.Run the Application:

```Bash
python main.py
```
4.PySimpleGUI Interface
<ul>
<li> Camera Feed: The main window displays the live camera feed with detected faces and recognized names.
<li> Chụp & Thêm Khuôn Mặt Button:
<ol> Type the employee's ID into the "Tên" input field.<br>
Click this button to capture the current frame and add the face to the system's database.</ol> 
<li>Gửi Check-In Button: Click this button to manually send a check-in request for the recognized person.
<li>Thoát Button: Click to close the application.
  </ul>
## Configuration

```known_faces_dir```: Path to the directory containing subfolders with known employee face images.<br>
```embeddings_file```: Path to the file where face embeddings will be stored.<br>
```recognition_threshold```: Distance threshold for face recognition (lower values are stricter).<br>
```device```: Set to 'cuda:0' if you have a CUDA-capable GPU, otherwise 'cpu'.<br>
```api_endpoint```: The URL of the API endpoint for check-ins. Replace with your actual endpoint.<br>
## Disclaimer
This project is intended for educational and demonstration purposes. The face recognition accuracy might vary depending on image quality, lighting conditions, and the chosen threshold.

## License
This project is licensed under the MIT License - see the LICENSE file for details.


<h1>Let me know if you'd like any adjustments to this README.md file!</h1>
