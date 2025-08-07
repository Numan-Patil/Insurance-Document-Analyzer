from app import db
from datetime import datetime

class Document(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    original_name = db.Column(db.String(255), nullable=False)
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)
    file_size = db.Column(db.Integer)
    page_count = db.Column(db.Integer)
    processed = db.Column(db.Boolean, default=False)
    
    def __repr__(self):
        return f'<Document {self.filename}>'

class Query(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    query_text = db.Column(db.Text, nullable=False)
    response_json = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    processing_time = db.Column(db.Float)
    
    def __repr__(self):
        return f'<Query {self.id}: {self.query_text[:50]}...>'
