# imports
from flask import Flask, render_template, request, redirect, url_for, jsonify, Response, stream_with_context
import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np
import tools
mp_drawing = mp.solutions.drawing_utils
# app
app = Flask(__name__)
global stop 
global desired_pose
desired_pose = ''
stop = False
#loading model
# model = tf.keras.models.load_model('./models/action-2.h5')

class Camera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.sequence = []
        self.posture = []
        self.predictions = []
        self.actions = np.array(['vrikshasana', 'tadasana', 'virabhadrasana'])
        self.threshold = 0.5
        self.current_action = ''
        
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
    
    def model_predict(self):
        model = tf.keras.models.load_model('./models/final-hope.h5')
        ret, frame = self.video.read()
        if ret:
            image, results = tools.mediapipe_detection(frame, self.pose)
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
            
            keypoints = tools.extract_keypoints(results)
            self.sequence.append(keypoints)
            self.sequence = self.sequence[-30:]
            
            if len(self.sequence) == 30:
                res = model.predict(np.expand_dims(self.sequence, axis=0))[0]
                print(self.actions[np.argmax(res)])
                self.predictions.append(np.argmax(res))
                
                #visualization
                if np.unique(self.predictions[-10:])[0] == np.argmax(res):
                    if res[np.argmax(res)] > self.threshold:
                        self.current_action = self.actions[np.argmax(res)]
                
                # image = tools.prob_viz(res, self.actions, image, colors=[(0, 255, 0) if x == np.argmax(res) else (0, 0, 255) for x in range(len(self.actions))])
            if(self.current_action == desired_pose):
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)        
            cv2.rectangle(image, (0, 0), (640, 40), (color), -1)
            cv2.putText(image, 'ACTION: ' + self.current_action, (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            # image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # image.flags.writeable = False
            # results = self.pose.process(image)
            # image.flags.writeable = True
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
            ret, jpeg = cv2.imencode('.jpg', image)
            return jpeg.tobytes()
        else:
            return None
# genereate frame
def gen(camera):
    while not stop:
        # frame = camera.get_frame()
        frame = camera.model_predict()
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
    global desired_pose
    if request.method == 'GET':
        return render_template('model_test.html', start=False, desired_pose=desired_pose)
    elif request.method == 'POST':
        desired_pose = request.form['desired_pose']
        return render_template('model_test.html', start=True, desired_pose=desired_pose)
        if request.form['stop'] == 'stop':
            stop = True
            return redirect(url_for('model_test'))
        
        


    
@app.route('/capture', methods=['GET', 'POST'])
def capture():
    return Response(stream_with_context(gen(Camera())), mimetype='multipart/x-mixed-replace; boundary=frame')
    
                
        
# run
if __name__ == '__main__':
    app.run(debug=True, port=6969)    