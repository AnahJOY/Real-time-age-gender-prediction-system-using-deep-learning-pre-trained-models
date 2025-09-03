import cv2
import numpy as np
from flask import Flask, render_template, Response, request, redirect, send_file, url_for
from PIL import Image
import io
import base64
import csv
from fpdf import FPDF
import tempfile

UPLOAD_FOLDER = './UPLOAD_FOLDER'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Global storage for prediction summaries
prediction_summary = []

# ----------------- Helper: Load Models -----------------
def load_models():
    faceProto = "opencv_face_detector.pbtxt"
    faceModel = "opencv_face_detector_uint8.pb"
    ageProto = "age_deploy.prototxt"
    ageModel = "age_net.caffemodel"
    genderProto = "gender_deploy.prototxt"
    genderModel = "gender_net.caffemodel"

    faceNet = cv2.dnn.readNet(faceModel, faceProto)
    ageNet = cv2.dnn.readNet(ageModel, ageProto)
    genderNet = cv2.dnn.readNet(genderModel, genderProto)

    return faceNet, ageNet, genderNet

# ----------------- Helper: Detect Face -----------------
def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight, frameWidth = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300),
                                 [104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2),
                          (0, 255, 0), int(round(frameHeight / 150)), 8)
    return frameOpencvDnn, faceBoxes

# ----------------- Live Streaming -----------------
def gen_frames():
    faceNet, ageNet, genderNet = load_models()
    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)',
               '(38-43)', '(48-53)', '(60-100)']
    genderList = ['Male', 'Female']
    video = cv2.VideoCapture(0)
    padding = 20

    while True:
        success, frame = video.read()
        if not success:
            break
        resultImg, faceBoxes = highlightFace(faceNet, frame)
        for faceBox in faceBoxes:
            face = frame[max(0, faceBox[1]-padding):min(faceBox[3]+padding, frame.shape[0]-1),
                         max(0, faceBox[0]-padding):min(faceBox[2]+padding, frame.shape[1]-1)]
            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
                                         MODEL_MEAN_VALUES, swapRB=False)
            gender = genderList[genderNet.setInput(blob) or genderNet.forward()[0].argmax()]
            age = ageList[ageNet.setInput(blob) or ageNet.forward()[0].argmax()]

            cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpg', resultImg)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# ----------------- One-shot Webcam Capture -----------------
def gen_frames_once():
    faceNet, ageNet, genderNet = load_models()
    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)',
               '(38-43)', '(48-53)', '(60-100)']
    genderList = ['Male', 'Female']
    video = cv2.VideoCapture(0)
    padding = 20

    while True:
        hasFrame, frame = video.read()
        if not hasFrame:
            break

        resultImg, faceBoxes = highlightFace(faceNet, frame)
        if not faceBoxes:
            continue

        for faceBox in faceBoxes:
            face = frame[max(0, faceBox[1]-padding):min(faceBox[3]+padding, frame.shape[0]-1),
                         max(0, faceBox[0]-padding):min(faceBox[2]+padding, frame.shape[1]-1)]

            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
                                         MODEL_MEAN_VALUES, swapRB=False)
            gender = genderList[genderNet.setInput(blob) or genderNet.forward()[0].argmax()]
            age = ageList[ageNet.setInput(blob) or ageNet.forward()[0].argmax()]

            cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

            ret, encodedImg = cv2.imencode('.jpg', resultImg)
            if ret:
                prediction_summary.append({
                    "source": "webcam",
                    "image": encodedImg.tobytes(),
                    "age": age,
                    "gender": gender
                })
                video.release()
                return True
    video.release()
    return False

# ----------------- Flask Routes -----------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/webcam')
def webcam():
    return render_template('webcam.html')

@app.route('/webcam_capture')
def webcam_capture():
    success = gen_frames_once()
    if success:
        return redirect(url_for("result"))
    return redirect(url_for("index"))

@app.route('/upload', methods=['POST'])
def upload_file():
    f = request.files['fileToUpload'].read()
    img = Image.open(io.BytesIO(f))
    img_ip = np.asarray(img, dtype="uint8")
    frame = cv2.cvtColor(img_ip, cv2.COLOR_BGR2RGB)

    faceNet, ageNet, genderNet = load_models()
    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)',
               '(38-43)', '(48-53)', '(60-100)']
    genderList = ['Male', 'Female']

    resultImg, faceBoxes = highlightFace(faceNet, frame)
    if faceBoxes:
        for faceBox in faceBoxes:
            face = frame[max(0, faceBox[1]-20):min(faceBox[3]+20, frame.shape[0]-1),
                         max(0, faceBox[0]-20):min(faceBox[2]+20, frame.shape[1]-1)]

            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
                                         MODEL_MEAN_VALUES, swapRB=False)
            gender = genderList[genderNet.setInput(blob) or genderNet.forward()[0].argmax()]
            age = ageList[ageNet.setInput(blob) or ageNet.forward()[0].argmax()]

            cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

            ret, encodedImg = cv2.imencode('.jpg', resultImg)
            if ret:
                prediction_summary.append({
                    "source": "photo",
                    "image": encodedImg.tobytes(),
                    "age": age,
                    "gender": gender
                })
    return redirect(url_for("result"))

@app.route('/result')
def result():
    if not prediction_summary:
        return redirect(url_for("index"))
    latest = prediction_summary[-1]
    latest_with_base64 = {
        "source": latest["source"],
        "age": latest["age"],
        "gender": latest["gender"],
        "image": base64.b64encode(latest["image"]).decode('utf-8')
    }
    return render_template('result.html', result=latest_with_base64)

@app.route('/summary')
def summary():
    summary_with_base64 = [
        {
            "source": item["source"],
            "age": item["age"],
            "gender": item["gender"],
            "image": base64.b64encode(item["image"]).decode('utf-8')
        } for item in prediction_summary
    ]
    return render_template('summary.html', summary=summary_with_base64)

@app.route('/clear_summary')
def clear_summary():
    prediction_summary.clear()
    return redirect('/summary')

@app.route('/download_csv')
def download_csv():
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, newline='', suffix='.csv') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Source', 'Gender', 'Age'])
        for item in prediction_summary:
            writer.writerow([item['source'], item['gender'], item['age']])
        csvfile.seek(0)
        return send_file(csvfile.name, as_attachment=True, download_name='detection_summary.csv')

@app.route('/download_pdf')
def download_pdf():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Detection Summary Report", ln=True, align='C')
    pdf.ln(10)

    for i, item in enumerate(prediction_summary, 1):
        pdf.cell(200, 10, txt=f"{i}. Source: {item['source']}, Gender: {item['gender']}, Age: {item['age']}", ln=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as pdf_file:
        pdf.output(pdf_file.name)
        return send_file(pdf_file.name, as_attachment=True, download_name='detection_summary.pdf')

if __name__ == '__main__':
    app.run(debug=True)
