# models.py
from flask_sqlalchemy import SQLAlchemy
import numpy as np


db = SQLAlchemy()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    photo_data = db.Column(db.String, nullable=False)
    face_encoding = db.Column(db.LargeBinary, nullable=True)
