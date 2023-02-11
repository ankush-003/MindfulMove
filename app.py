# imports
from flask import Flask, render_template, request, redirect, url_for, jsonify, Response, stream_with_context
import cv2
import mediapipe as mp
import tensorflow as tf
mp_drawing = mp.solutions.drawing_utils
# app
app = Flask(__name__)
global stop 
stop = False
#loading model
model = tf.keras.models.load_model('./models/action-2.h5')

class Camera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
    def __del__(self):
        self.video.release()
        
    def get_frame(self):
        ret, frame = self.video.read()
        if ret:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = self.pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
            ret, jpeg = cv2.imencode('.jpg', image)
            return jpeg.tobytes()
        else:
            return None
        
# genereate frame
def gen(camera):
    while not stop:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    camera.__del__()        

# routes


@app.route('/', methods=['GET'])
def home():
    if request.method == 'GET':
        return render_template('home.html')
   
@app.route('/exercise', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    elif request.method == 'POST':
        if request.form['stop'] == 'stop':
            stop = True
            return redirect(url_for('index'))


@app.route('/model_test', methods=['GET', 'POST'])
def model():
    if request.method == 'GET':
        return render_template('model_test.html')
    elif request.method == 'POST':
        if request.form['stop'] == 'stop':
            stop = True
            return redirect(url_for('model_test'))


    
@app.route('/capture', methods=['GET', 'POST'])
def capture():
    return Response(stream_with_context(gen(Camera())), mimetype='multipart/x-mixed-replace; boundary=frame')
    
                
        
# run
if __name__ == '__main__':
    app.run(debug=True, port=6969)    