from flask import Flask, render_template, redirect, url_for, flash,jsonify,session
from flask_wtf import FlaskForm
from wtforms import SubmitField
import os
import cv2
import threading
import base64
import face_recognition
from io import BytesIO
from models import db, User
import numpy as np
import start
import check


app = Flask(__name__)
global id
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///facerecog.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

known_faces_folder = 'known_faces'
uploads_folder = 'uploads'

known_face_image_path = os.path.join(known_faces_folder, 'known_face.jpg')
known_face_encoding = face_recognition.face_encodings(face_recognition.load_image_file(known_face_image_path))[0]

class RegistrationForm(FlaskForm):
    submit = SubmitField('Register')

class LoginForm(FlaskForm):
    submit = SubmitField('Login')

def capture_photo():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if not ret:
        raise ValueError('Error finding the frame')
    cap.release()

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    _, buffer = cv2.imencode('.jpg', rgb_frame)
    image_as_str = base64.b64encode(buffer).decode('utf-8')

    return image_as_str

def compare_face_encoding(saved_encoding, unknown_encoding):
    results = face_recognition.compare_faces([saved_encoding], unknown_encoding)
    return results[0]

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    try:
        if form.validate_on_submit():
            # Capture the photo data and encode the face
            photo_data = capture_photo()
            unknown_face_encoding = face_recognition.face_encodings(
                face_recognition.load_image_file(BytesIO(base64.b64decode(photo_data)))
            )

            if not unknown_face_encoding:
                flash('No face detected in the captured photo. Please try again.', 'danger')
                return redirect(url_for('register'))

            # Retrieve all users from the database
            users = User.query.all()

            for user in users:
                # Retrieve the face encoding stored during registration
                saved_user_encoding = np.frombuffer(user.face_encoding) if user.face_encoding else None

                if saved_user_encoding is not None:
                    # Compare the face encodings
                    result = compare_face_encoding(saved_user_encoding, unknown_face_encoding[0])

                    if result:
                        flash('User already registered.', 'danger')
                        return redirect(url_for('login'))

            # If the user is not found in the database, register the new user
            new_user = User(photo_data=photo_data, face_encoding=unknown_face_encoding[0].tobytes())
            db.session.add(new_user)
            db.session.commit()

            flash('Registration successful!', 'success')
            return redirect(url_for('login'))
    except Exception as e:
        flash(f'Error occurred during registration: {str(e)}', 'danger')

    return render_template('register.html', form=form)



@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    try:
        if form.validate_on_submit():
            print("in form")
            photo_data = capture_photo()
            unknown_face_encoding = face_recognition.face_encodings(
                face_recognition.load_image_file(BytesIO(base64.b64decode(photo_data)))
            )

            if not unknown_face_encoding:
                print("face not recognized")
                flash('No face detected in the captured photo. Please try again.', 'danger')
                return redirect(url_for('login'))

            users = User.query.all()

            for user in users:
                print("finding user")
                saved_user_encoding = np.frombuffer(user.face_encoding) if user.face_encoding else None

                if saved_user_encoding is not None:
                    result = compare_face_encoding(saved_user_encoding, unknown_face_encoding[0])

                    if result:
                        print("finding similarity")
                        session['user_id'] = user.id
                        # flash('Login successful!', 'success')
                        return render_template('start.html')

            flash('Face not recognized. Please try again.', 'danger')

    except Exception as e:
        flash(f'An error occurred during login: {str(e)}', 'danger')

    return render_template('login.html',form=form)

@app.route('/start')
def started():
    start.start()
    return "exam completed"

@app.route('/finished')
def finished():
    return render_template('started.html')

@app.route('/logout')
def logout():
    session.pop('user_id',None)
    flash(f'You have been logged out')
    return redirect(url_for('login'))
    
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    print(id)
    app.run(debug=True)
