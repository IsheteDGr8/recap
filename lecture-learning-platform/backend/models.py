from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text, JSON, Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import bcrypt

Base = declarative_base()

# Association tables for many-to-many relationships
study_group_members = Table(
    'study_group_members',
    Base.metadata,
    Column('user_id', Integer, ForeignKey('users.id')),
    Column('group_id', Integer, ForeignKey('study_groups.id'))
)

lecture_shares = Table(
    'lecture_shares',
    Base.metadata,
    Column('lecture_id', Integer, ForeignKey('lectures.id')),
    Column('group_id', Integer, ForeignKey('study_groups.id'))
)

class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    lectures = relationship("Lecture", back_populates="owner")
    progress = relationship("LectureProgress", back_populates="user")
    flashcard_progress = relationship("FlashcardProgress", back_populates="user")
    owned_groups = relationship("StudyGroup", back_populates="creator")
    groups = relationship("StudyGroup", secondary=study_group_members, back_populates="members")
    
    def set_password(self, password):
        self.password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    def check_password(self, password):
        return bcrypt.checkpw(password.encode('utf-8'), self.password_hash.encode('utf-8'))

class Lecture(Base):
    __tablename__ = 'lectures'
    
    id = Column(Integer, primary_key=True)
    title = Column(String(255), nullable=False)
    description = Column(Text)
    video_source = Column(String(20))  # 'upload' or 'url'
    video_url = Column(String(500))  # Original URL if from link
    duration = Column(Float)
    transcript = Column(Text)
    transcript_with_timestamps = Column(JSON)
    summary = Column(Text)
    key_points = Column(JSON)
    quiz = Column(JSON)
    study_plan = Column(JSON)
    practice_problems = Column(JSON)
    flashcards = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Foreign keys
    owner_id = Column(Integer, ForeignKey('users.id'))
    
    # Relationships
    owner = relationship("User", back_populates="lectures")
    progress = relationship("LectureProgress", back_populates="lecture")
    shared_with_groups = relationship("StudyGroup", secondary=lecture_shares, back_populates="shared_lectures")

class LectureProgress(Base):
    __tablename__ = 'lecture_progress'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    lecture_id = Column(Integer, ForeignKey('lectures.id'))
    
    # Progress tracking
    last_position = Column(Float, default=0.0)  # Last watched position in seconds
    completed = Column(Boolean, default=False)
    quiz_taken = Column(Boolean, default=False)
    quiz_score = Column(Integer)
    quiz_attempts = Column(Integer, default=0)
    notes = Column(Text)
    
    # Study plan tracking
    study_plan_progress = Column(JSON)  # Track completed days/tasks
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="progress")
    lecture = relationship("Lecture", back_populates="progress")

class FlashcardProgress(Base):
    __tablename__ = 'flashcard_progress'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    lecture_id = Column(Integer, ForeignKey('lectures.id'))
    flashcard_index = Column(Integer)
    
    # Spaced repetition data
    ease_factor = Column(Float, default=2.5)
    interval = Column(Integer, default=0)  # Days until next review
    repetitions = Column(Integer, default=0)
    next_review = Column(DateTime, default=datetime.utcnow)
    last_reviewed = Column(DateTime)
    
    # Performance tracking
    times_seen = Column(Integer, default=0)
    times_correct = Column(Integer, default=0)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="flashcard_progress")

class StudyGroup(Base):
    __tablename__ = 'study_groups'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    invite_code = Column(String(20), unique=True)
    creator_id = Column(Integer, ForeignKey('users.id'))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    creator = relationship("User", back_populates="owned_groups")
    members = relationship("User", secondary=study_group_members, back_populates="groups")
    shared_lectures = relationship("Lecture", secondary=lecture_shares, back_populates="shared_with_groups")

class GroupMessage(Base):
    __tablename__ = 'group_messages'
    
    id = Column(Integer, primary_key=True)
    group_id = Column(Integer, ForeignKey('study_groups.id'))
    user_id = Column(Integer, ForeignKey('users.id'))
    message = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)