# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 16:26:43 2022

@author: xurui
"""

from flask import Flask,render_template,request,redirect,url_for,make_response,jsonify, Response
import os,cv2,time
from datetime import timedelta
import numpy as np
from track1 import run
# from track import run

# run(source = 'D:/research/Iot/video/vehicle/Toyota_rav4_2015_black_03.mp4')#, save_vid=True)

upload_path=''
ALLOWED_EXTENSIONS = {'png', 'jpg', 'JPG', 'PNG', 'bmp','mp4','ts'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

app = Flask(__name__, template_folder='templates')

app.SEND_FILE_MAX_AGE_DEFAULT = timedelta(seconds=1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/to_upload_page', methods=['POST', 'GET'])
def to_upload_page():
    if request.method == 'POST':
        f = request.files['file']
        if not (f and allowed_file(f.filename)):
            return jsonify({"error": 1001, "msg": "Please check file type, mp4 only"})

        user_input = request.form.get("name")
        print('user_input:')
        print(user_input)

        basepath = os.path.dirname(__file__)
        global upload_path  #
        upload_path = os.path.join(basepath, 'video',f.filename)  # create folder before this
        print ("name is %s, path is %s"%(f.filename,upload_path))
        f.save(upload_path)
        print('file saved to', upload_path)
        print ("finished upload")
        return render_template('upload_ok_video.html', userinput=user_input, val1=time.time())
    return  render_template('upload.html')


@app.route('/video_feed')
def video_feed():
    print(upload_path)
    return Response(run(upload_path), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/video1')
def video1():
    return render_template('video1.html')
@app.route('/video1_v')
def video1_v():
    print('video1')
    return  Response(run('D:/research/drone/UAV-benchmark-M/M0101/M0101.mp4'), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/realtime')
def realtime():
    return render_template('realtime.html')
@app.route('/webcam1')
def webcam1():
    print('webcam')
    return  Response(run(0), mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__ == '__main__':
    #app.run(debug=True, use_reloader=False)
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)# 0.0.0.0


#%% method 2 save the vid, display
run(source = 'D:/research/Iot/video/vehicle/BMW_320I_2018_black_3.MOV', save_vid = True, save_file_name = 'test1.mp4')

cap = cv2.VideoCapture('test1.mp4')
print( cap.isOpened())
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # int width
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # int height
print(f'width: {width}, height: {height}')
print(fps)

while True:
    ret, frame = cap.read()
    if ret:
        #frame = cv2.resize(frame, size)
        # convert img to bytes
        imgencode=cv2.imencode('.jpg',frame)[1]
        stringData=imgencode.tobytes()
        yield (b'--frame\r\n'
            b'Content-Type: text/plain\r\n\r\n'+stringData+b'\r\n')
        #cv2.imshow("A video", frame)
        #c = cv2.waitKey(100)
        #if c == 27:
        #    break
    else:
        print('camera.release()')
        cap.release()
        break
cap.release()
cv2.destroyAllWindows()

