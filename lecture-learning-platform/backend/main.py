import os
from dotenv import load_dotenv

load_dotenv()

# Determine AI provider
AI_PROVIDER = os.getenv("AI_PROVIDER", "openai").lower()

# Initialize clients based on provider
if AI_PROVIDER == "twelvelabs":
    from twelvelabs import TwelveLabs
    twelvelabs_client = TwelveLabs(api_key=os.getenv("TWELVELABS_API_KEY"))
    TWELVELABS_INDEX_ID = os.getenv("TWELVELABS_INDEX_ID")
    print(f"🎥 Using Twelve Labs for video processing")
    print(f"   Index ID: {TWELVELABS_INDEX_ID}")

import openai
openai.api_key = os.getenv("OPENAI_API_KEY")
print(f"🤖 Using OpenAI GPT-4 for content generation")

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from typing import List, Optional
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import json
import tempfile
import subprocess
from pathlib import Path
import uvicorn
import secrets
import yt_dlp
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import io
import time

from database import get_db, init_db
from models import User, Lecture, LectureProgress, FlashcardProgress, StudyGroup, GroupMessage
from auth import create_access_token, get_current_user, ACCESS_TOKEN_EXPIRE_MINUTES

app = FastAPI(title="Lecture Learning Platform - Advanced")
init_db()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============== Pydantic Models (same as before) ==============
# ... [Copy all Pydantic models from original main.py] ...

class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str

class TimestampSummaryRequest(BaseModel):
    lecture_id: int
    start_time: float = 0.0
    end_time: Optional[float] = None

class QuizSubmit(BaseModel):
    lecture_id: int
    answers: List[int]

class ProgressUpdate(BaseModel):
    lecture_id: int
    last_position: Optional[float] = None
    completed: Optional[bool] = None
    notes: Optional[str] = None

class FlashcardReview(BaseModel):
    lecture_id: int
    flashcard_index: int
    quality: int

class GroupCreate(BaseModel):
    name: str
    description: Optional[str] = None

class ExportRequest(BaseModel):
    lecture_id: int
    format: str

# ============== Twelve Labs Functions ==============

def transcribe_with_twelvelabs(video_path: str) -> dict:
    """Transcribe video using Twelve Labs API"""
    try:
        print("Uploading video to Twelve Labs...", flush=True)
        
        # Upload video
        task = twelvelabs_client.task.create(
            index_id=TWELVELABS_INDEX_ID,
            file=video_path,
            language="en"
        )
        
        print(f"Task created: {task.id}", flush=True)
        print("Waiting for processing...", flush=True)
        
        # Wait for processing (can take 1-5 minutes)
        def on_task_update(task):
            print(f"Progress: {task.status}", flush=True)
        
        task.wait_for_done(
            sleep_interval=5,
            callback=on_task_update
        )
        
        if task.status != "ready":
            raise Exception(f"Task failed with status: {task.status}")
        
        video_id = task.video_id
        print(f"Video processed! ID: {video_id}", flush=True)
        
        # Get transcript
        print("Fetching transcript...", flush=True)
        transcript_result = twelvelabs_client.generate.text(
            video_id=video_id,
            prompt="Provide a complete, accurate transcript of this video with speaker labels and timestamps."
        )
        
        # Get video info for duration
        video_info = twelvelabs_client.index.video.get(
            index_id=TWELVELABS_INDEX_ID,
            id=video_id
        )
        
        duration = video_info.metadata.duration if hasattr(video_info.metadata, 'duration') else 0
        
        # Parse transcript into segments
        # Twelve Labs returns formatted text, we'll create simple segments
        full_text = transcript_result.data
        
        # Create simple segments (we can enhance this later)
        segments = [{
            "start": 0,
            "end": duration,
            "text": full_text
        }]
        
        return {
            "full_text": full_text,
            "segments": segments,
            "duration": duration,
            "video_id": video_id  # Store for future reference
        }
        
    except Exception as e:
        print(f"Twelve Labs error: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=f"Twelve Labs transcription failed: {str(e)}")

def transcribe_with_openai(file_path: str) -> dict:
    """Transcribe audio using OpenAI Whisper API"""
    try:
        with open(file_path, 'rb') as audio_file:
            transcript = openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="verbose_json",
                timestamp_granularities=["segment"]
            )
        
        segments = []
        full_text = []
        
        for segment in transcript.segments:
            segments.append({
                "start": segment['start'] if isinstance(segment, dict) else segment.start,
                "end": segment['end'] if isinstance(segment, dict) else segment.end,
                "text": segment['text'] if isinstance(segment, dict) else segment.text
            })
            full_text.append(segment['text'] if isinstance(segment, dict) else segment.text)
        
        duration = transcript.duration if hasattr(transcript, 'duration') else (segments[-1]['end'] if segments else 0)
        
        return {
            "full_text": " ".join(full_text),
            "segments": segments,
            "duration": duration
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI transcription failed: {str(e)}")

def transcribe_video(video_path: str) -> dict:
    """Transcribe video using selected provider"""
    if AI_PROVIDER == "twelvelabs":
        return transcribe_with_twelvelabs(video_path)
    else:
        # Extract audio for OpenAI
        audio_path = video_path.replace(Path(video_path).suffix, '.wav')
        extract_audio(video_path, audio_path)
        
        transcribe_path = audio_path if os.path.exists(audio_path) else video_path
        result = transcribe_with_openai(transcribe_path)
        
        # Cleanup audio file
        if os.path.exists(audio_path):
            os.remove(audio_path)
        
        return result

# ============== Helper Functions ==============

def extract_audio(video_path: str, audio_path: str):
    """Extract audio from video (only needed for OpenAI)"""
    try:
        subprocess.run([
            'ffmpeg', '-i', video_path,
            '-vn', '-acodec', 'pcm_s16le',
            '-ar', '16000', '-ac', '1',
            audio_path
        ], check=True, capture_output=True)
        return True
    except:
        return False

def download_video_from_url(url: str, output_path: str) -> str:
    """Download video from URL"""
    ydl_opts = {
        'format': 'best[ext=mp4]/best',
        'outtmpl': output_path,
        'quiet': True,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return ydl.prepare_filename(info)

def get_transcript_by_timerange(segments: List[dict], start_min: float, end_min: Optional[float]) -> str:
    start_sec = start_min * 60
    end_sec = end_min * 60 if end_min else float('inf')
    
    filtered_segments = [
        seg['text'] for seg in segments
        if seg['start'] >= start_sec and seg['start'] <= end_sec
    ]
    
    return " ".join(filtered_segments)

def generate_summary(transcript: str, start_time: float = 0, end_time: Optional[float] = None) -> dict:
    """Generate summary using OpenAI GPT-4"""
    time_context = ""
    if start_time > 0 or end_time:
        end_str = f"{end_time:.1f}" if end_time else "end"
        time_context = f"\n\nNote: Time range {start_time:.1f} to {end_str} minutes."
    
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert educational content analyzer."},
            {"role": "user", "content": f"""Analyze this lecture and provide:
1. A comprehensive summary (3-4 paragraphs)
2. 5-7 key points as a JSON array
{time_context}

Transcript: {transcript[:4000]}

Respond in JSON format:
{{
    "summary": "...",
    "key_points": ["point 1", "point 2", ...]
}}"""}
        ],
        response_format={"type": "json_object"}
    )
    
    return json.loads(response.choices[0].message.content)

# [Copy all other generation functions: generate_quiz, generate_study_plan, generate_practice_problems, generate_flashcards, etc.]

def generate_quiz(transcript: str) -> List[dict]:
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert at creating educational assessments."},
            {"role": "user", "content": f"""Create 5 multiple-choice quiz questions based on this lecture.

Transcript: {transcript[:4000]}

Respond in JSON format:
{{
    "questions": [
        {{
            "question": "Question text?",
            "options": ["Option A", "Option B", "Option C", "Option D"],
            "correct_answer": 0,
            "explanation": "Why this is correct"
        }}
    ]
}}"""}
        ],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)["questions"]

def generate_study_plan(transcript: str, key_points: List[str]) -> dict:
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert educational planner."},
            {"role": "user", "content": f"""Create a 7-day study plan for this lecture.

Key Points: {json.dumps(key_points)}

Respond in JSON format:
{{
    "overview": "Study plan overview",
    "days": [
        {{
            "day": 1,
            "focus": "Topic",
            "tasks": ["Task 1", "Task 2"],
            "duration": "30-45 minutes"
        }}
    ]
}}"""}
        ],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

def generate_practice_problems(transcript: str) -> List[dict]:
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert at creating practice problems."},
            {"role": "user", "content": f"""Create 5 practice problems.

Transcript: {transcript[:4000]}

Respond in JSON format:
{{
    "problems": [
        {{
            "problem": "Problem statement",
            "difficulty": "easy/medium/hard",
            "hint": "A helpful hint",
            "solution": "Step-by-step solution"
        }}
    ]
}}"""}
        ],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)["problems"]

def generate_flashcards(transcript: str, key_points: List[str]) -> List[dict]:
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert at creating flashcards."},
            {"role": "user", "content": f"""Create 10 flashcards.

Key Points: {json.dumps(key_points)}

Respond in JSON format:
{{
    "flashcards": [
        {{
            "front": "Question",
            "back": "Answer",
            "category": "topic"
        }}
    ]
}}"""}
        ],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)["flashcards"]

def calculate_next_review(quality: int, ease_factor: float, interval: int, repetitions: int):
    """SM-2 Algorithm"""
    if quality < 3:
        interval = 0
        repetitions = 0
    else:
        if repetitions == 0:
            interval = 1
        elif repetitions == 1:
            interval = 6
        else:
            interval = round(interval * ease_factor)
        repetitions += 1
    
    ease_factor = ease_factor + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
    if ease_factor < 1.3:
        ease_factor = 1.3
    
    next_review = datetime.utcnow() + timedelta(days=interval)
    return ease_factor, interval, repetitions, next_review

# ============== Upload Endpoint ==============

@app.post("/api/upload-lecture")
async def upload_lecture(
    file: Optional[UploadFile] = File(None),
    title: str = Form(...),
    description: Optional[str] = Form(None),
    video_url: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    import sys
    import traceback
    
    video_path = None
    
    try:
        print("\n" + "="*50, file=sys.stderr)
        print(f"Provider: {AI_PROVIDER}", file=sys.stderr)
        print(f"User: {current_user.username}", file=sys.stderr)
        print(f"Title: {title}", file=sys.stderr)
        print("="*50 + "\n", file=sys.stderr)
        
        if file:
            print("Processing file upload...", file=sys.stderr)
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
                content = await file.read()
                tmp_file.write(content)
                video_path = tmp_file.name
            video_source = "upload"
            source_url = None
        elif video_url:
            print("Processing URL download...", file=sys.stderr)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                video_path = tmp_file.name
            video_path = download_video_from_url(video_url, video_path)
            video_source = "url"
            source_url = video_url
        else:
            raise HTTPException(status_code=400, detail="File or URL required")
        
        print(f"Video saved to: {video_path}", file=sys.stderr)
        
        # Transcribe using selected provider
        print(f"Transcribing with {AI_PROVIDER}...", file=sys.stderr)
        transcript_data = transcribe_video(video_path)
        print("Transcription complete!", file=sys.stderr)
        
        # Generate content with GPT-4
        print("Generating summary...", file=sys.stderr)
        summary_data = generate_summary(transcript_data['full_text'])
        
        print("Generating quiz...", file=sys.stderr)
        quiz = generate_quiz(transcript_data['full_text'])
        
        print("Generating study plan...", file=sys.stderr)
        study_plan = generate_study_plan(transcript_data['full_text'], summary_data['key_points'])
        
        print("Generating practice problems...", file=sys.stderr)
        practice_problems = generate_practice_problems(transcript_data['full_text'])
        
        print("Generating flashcards...", file=sys.stderr)
        flashcards = generate_flashcards(transcript_data['full_text'], summary_data['key_points'])
        
        # Save to database
        print("Saving to database...", file=sys.stderr)
        lecture = Lecture(
            title=title,
            description=description,
            video_source=video_source,
            video_url=source_url,
            duration=transcript_data['duration'],
            transcript=transcript_data['full_text'],
            transcript_with_timestamps=transcript_data['segments'],
            summary=summary_data['summary'],
            key_points=summary_data['key_points'],
            quiz=quiz,
            study_plan=study_plan,
            practice_problems=practice_problems,
            flashcards=flashcards,
            owner_id=current_user.id
        )
        
        db.add(lecture)
        db.commit()
        db.refresh(lecture)
        
        print("SUCCESS!", file=sys.stderr)
        return lecture
        
    except Exception as e:
        print("\n" + "!"*50, file=sys.stderr)
        print("ERROR:", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        print("!"*50 + "\n", file=sys.stderr)
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        if video_path and os.path.exists(video_path):
            os.remove(video_path)

# [Copy all other endpoints from original main.py: auth, lectures, progress, quiz, flashcards, groups, export]

# ============== Auth Endpoints ==============

@app.post("/api/auth/register")
async def register(user: UserCreate, db: Session = Depends(get_db)):
    if db.query(User).filter(User.username == user.username).first():
        raise HTTPException(status_code=400, detail="Username exists")
    if db.query(User).filter(User.email == user.email).first():
        raise HTTPException(status_code=400, detail="Email exists")
    
    new_user = User(username=user.username, email=user.email)
    new_user.set_password(user.password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    access_token = create_access_token(
        data={"sub": new_user.username},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "username": new_user.username,
        "email": new_user.email
    }

@app.post("/api/auth/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not user.check_password(form_data.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    access_token = create_access_token(
        data={"sub": user.username},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "username": user.username,
        "email": user.email
    }

@app.get("/api/auth/me")
async def get_me(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return {"id": current_user.id, "username": current_user.username, "email": current_user.email}

@app.get("/api/lectures")
async def get_lectures(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    lectures = db.query(Lecture).filter(Lecture.owner_id == current_user.id).all()
    return lectures

@app.get("/")
async def root():
    return {
        "message": "Lecture Learning Platform API",
        "status": "running",
        "ai_provider": AI_PROVIDER
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)