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

# ============== Pydantic Models ==============

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

def transcribe_with_twelvelabs(video_path: str = None, video_url: str = None) -> dict:
    """Transcribe video using Twelve Labs API - supports both files and URLs"""
    try:
        print("Processing with Twelve Labs...", flush=True)
        
        # IMPORTANT: Twelve Labs can process URLs directly!
        if video_url:
            print(f"✅ Using URL directly: {video_url}", flush=True)
            task = twelvelabs_client.task.create(
                index_id=TWELVELABS_INDEX_ID,
                url=video_url,  # Pass URL directly - no download needed!
                language="en"
            )
        elif video_path:
            # Check file size before uploading
            file_size = os.path.getsize(video_path) / (1024 * 1024)  # Convert to MB
            print(f"File size: {file_size:.2f} MB", flush=True)
            
            if file_size > 500:  # 500MB safety limit
                raise Exception(
                    f"File too large ({file_size:.0f}MB). "
                    f"Please use videos under 500MB or provide a YouTube/Vimeo URL instead."
                )
            
            print("Uploading file to Twelve Labs...", flush=True)
            with open(video_path, 'rb') as file:
                task = twelvelabs_client.task.create(
                    index_id=TWELVELABS_INDEX_ID,
                    file=file,
                    language="en"
                )
        else:
            raise Exception("Either video_path or video_url must be provided")
        
        print(f"Task created: {task.id}", flush=True)
        print("Processing video (this may take 1-5 minutes depending on length)...", flush=True)
        
        # Wait for Twelve Labs to process
        def on_task_update(task):
            print(f"  Status: {task.status}", flush=True)
        
        task.wait_for_done(
            sleep_interval=10,
            callback=on_task_update
        )
        
        if task.status != "ready":
            raise Exception(f"Task failed with status: {task.status}")
        
        video_id = task.video_id
        print(f"✅ Video processed! ID: {video_id}", flush=True)
        
        # Get comprehensive transcript with timestamps
        print("Fetching detailed transcript...", flush=True)
        transcript_result = twelvelabs_client.generate.text(
            video_id=video_id,
            prompt="Provide a complete, word-for-word transcript of this video. Include all spoken content in detail."
        )
        
        # Get video metadata
        video_info = twelvelabs_client.index.video.get(
            index_id=TWELVELABS_INDEX_ID,
            id=video_id
        )
        
        duration = video_info.metadata.duration if hasattr(video_info.metadata, 'duration') else 0
        full_text = transcript_result.data
        
        # Create basic segments (Twelve Labs doesn't provide detailed timestamps in free tier)
        # We'll create segments based on duration
        segments = []
        if duration > 0:
            # Split into roughly 5-minute segments for better time-based summaries
            segment_duration = 300  # 5 minutes in seconds
            num_segments = max(1, int(duration / segment_duration))
            segment_length = len(full_text) // num_segments
            
            for i in range(num_segments):
                start_time = i * segment_duration
                end_time = min((i + 1) * segment_duration, duration)
                start_char = i * segment_length
                end_char = (i + 1) * segment_length if i < num_segments - 1 else len(full_text)
                
                segments.append({
                    "start": start_time,
                    "end": end_time,
                    "text": full_text[start_char:end_char].strip()
                })
        else:
            # Fallback: single segment
            segments = [{
                "start": 0,
                "end": 0,
                "text": full_text
            }]
        
        print(f"✅ Created {len(segments)} transcript segments", flush=True)
        
        return {
            "full_text": full_text,
            "segments": segments,
            "duration": duration,
            "video_id": video_id
        }
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Twelve Labs error: {error_msg}", flush=True)
        raise HTTPException(status_code=500, detail=f"Twelve Labs transcription failed: {error_msg}")


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


def transcribe_video(video_path: str = None, video_url: str = None) -> dict:
    """Transcribe video using selected AI provider"""
    if AI_PROVIDER == "twelvelabs":
        return transcribe_with_twelvelabs(video_path=video_path, video_url=video_url)
    else:
        # OpenAI requires audio extraction
        if not video_path:
            raise HTTPException(status_code=400, detail="OpenAI provider requires a video file")
        
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
    """Download video from URL (only used for OpenAI provider)"""
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
        
        # Validate input
        if not file and not video_url:
            raise HTTPException(status_code=400, detail="Either file or video_url must be provided")
        
        # For Twelve Labs with URLs - skip download entirely!
        if AI_PROVIDER == "twelvelabs" and video_url:
            print("✅ Using Twelve Labs with URL - no download needed!", file=sys.stderr)
            transcript_data = transcribe_video(video_url=video_url)
            video_source = "url"
            source_url = video_url
            
        # For file uploads
        elif file:
            print("Processing file upload...", file=sys.stderr)
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
                content = await file.read()
                tmp_file.write(content)
                video_path = tmp_file.name
            
            print(f"Video saved to: {video_path}", file=sys.stderr)
            transcript_data = transcribe_video(video_path=video_path)
            video_source = "upload"
            source_url = None
            
        # For OpenAI with URLs - need to download
        elif video_url:
            print("Downloading video from URL...", file=sys.stderr)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                video_path = tmp_file.name
            video_path = download_video_from_url(video_url, video_path)
            
            print(f"Video downloaded to: {video_path}", file=sys.stderr)
            transcript_data = transcribe_video(video_path=video_path)
            video_source = "url"
            source_url = video_url
        
        # Generate educational content with OpenAI (always)
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
        
        # Create lecture in database
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
        
        print("✅ SUCCESS! Lecture saved.", file=sys.stderr)
        print("="*50 + "\n", file=sys.stderr)
        
        return lecture
        
    except Exception as e:
        print("\n" + "!"*50, file=sys.stderr)
        print("ERROR:", file=sys.stderr)
        print(f"  Type: {type(e).__name__}", file=sys.stderr)
        print(f"  Message: {str(e)}", file=sys.stderr)
        print("\nTraceback:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        print("!"*50 + "\n", file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}")
        
    finally:
        # Cleanup temporary files
        if video_path and os.path.exists(video_path):
            try:
                os.remove(video_path)
                print(f"Cleaned up: {video_path}", file=sys.stderr)
            except:
                pass

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
async def get_me(current_user: User = Depends(get_current_user)):
    return {"id": current_user.id, "username": current_user.username, "email": current_user.email}

@app.get("/api/lectures")
async def get_lectures(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    lectures = db.query(Lecture).filter(Lecture.owner_id == current_user.id).all()
    return lectures

@app.get("/api/lectures/{lecture_id}")
async def get_lecture(lecture_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    lecture = db.query(Lecture).filter(Lecture.id == lecture_id, Lecture.owner_id == current_user.id).first()
    if not lecture:
        raise HTTPException(status_code=404, detail="Lecture not found")
    return lecture

@app.post("/api/timestamp-summary")
async def get_timestamp_summary(
    request: TimestampSummaryRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    lecture = db.query(Lecture).filter(
        Lecture.id == request.lecture_id,
        Lecture.owner_id == current_user.id
    ).first()
    
    if not lecture:
        raise HTTPException(status_code=404, detail="Lecture not found")
    
    # Get transcript for time range
    transcript_segment = get_transcript_by_timerange(
        lecture.transcript_with_timestamps,
        request.start_time,
        request.end_time
    )
    
    # Generate summary for this time range
    summary_data = generate_summary(transcript_segment, request.start_time, request.end_time)
    
    return {
        "start_time": request.start_time,
        "end_time": request.end_time,
        "summary": summary_data["summary"],
        "key_points": summary_data["key_points"]
    }

@app.post("/api/progress")
async def update_progress(
    progress: ProgressUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    lecture_progress = db.query(LectureProgress).filter(
        LectureProgress.lecture_id == progress.lecture_id,
        LectureProgress.user_id == current_user.id
    ).first()
    
    if not lecture_progress:
        lecture_progress = LectureProgress(
            lecture_id=progress.lecture_id,
            user_id=current_user.id
        )
        db.add(lecture_progress)
    
    if progress.last_position is not None:
        lecture_progress.last_position = progress.last_position
    if progress.completed is not None:
        lecture_progress.completed = progress.completed
    if progress.notes is not None:
        lecture_progress.notes = progress.notes
    
    lecture_progress.last_accessed = datetime.utcnow()
    
    db.commit()
    db.refresh(lecture_progress)
    return lecture_progress

@app.get("/api/progress/{lecture_id}")
async def get_progress(
    lecture_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    progress = db.query(LectureProgress).filter(
        LectureProgress.lecture_id == lecture_id,
        LectureProgress.user_id == current_user.id
    ).first()
    
    if not progress:
        return {"lecture_id": lecture_id, "last_position": 0, "completed": False, "notes": ""}
    
    return progress

@app.post("/api/submit-quiz")
async def submit_quiz(
    submission: QuizSubmit,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    lecture = db.query(Lecture).filter(
        Lecture.id == submission.lecture_id,
        Lecture.owner_id == current_user.id
    ).first()
    
    if not lecture:
        raise HTTPException(status_code=404, detail="Lecture not found")
    
    results = []
    correct_count = 0
    
    for i, answer in enumerate(submission.answers):
        if i < len(lecture.quiz):
            is_correct = answer == lecture.quiz[i]["correct_answer"]
            if is_correct:
                correct_count += 1
            
            results.append({
                "question_index": i,
                "user_answer": answer,
                "correct_answer": lecture.quiz[i]["correct_answer"],
                "is_correct": is_correct,
                "explanation": lecture.quiz[i]["explanation"]
            })
    
    score = (correct_count / len(lecture.quiz)) * 100 if lecture.quiz else 0
    
    # Update progress
    progress = db.query(LectureProgress).filter(
        LectureProgress.lecture_id == submission.lecture_id,
        LectureProgress.user_id == current_user.id
    ).first()
    
    if progress:
        progress.quiz_attempts = (progress.quiz_attempts or 0) + 1
        progress.quiz_score = score
        db.commit()
    
    return {
        "score": score,
        "correct": correct_count,
        "total": len(lecture.quiz),
        "results": results
    }

@app.get("/api/flashcards/{lecture_id}")
async def get_flashcards(
    lecture_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    lecture = db.query(Lecture).filter(
        Lecture.id == lecture_id,
        Lecture.owner_id == current_user.id
    ).first()
    
    if not lecture:
        raise HTTPException(status_code=404, detail="Lecture not found")
    
    return {"flashcards": lecture.flashcards}

@app.get("/api/flashcards/due/{lecture_id}")
async def get_due_flashcards(
    lecture_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    now = datetime.utcnow()
    due_cards = db.query(FlashcardProgress).filter(
        FlashcardProgress.lecture_id == lecture_id,
        FlashcardProgress.user_id == current_user.id,
        FlashcardProgress.next_review <= now
    ).all()
    
    return {"due_count": len(due_cards), "cards": due_cards}

@app.post("/api/flashcards/review")
async def review_flashcard(
    review: FlashcardReview,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    progress = db.query(FlashcardProgress).filter(
        FlashcardProgress.lecture_id == review.lecture_id,
        FlashcardProgress.user_id == current_user.id,
        FlashcardProgress.flashcard_index == review.flashcard_index
    ).first()
    
    if not progress:
        progress = FlashcardProgress(
            lecture_id=review.lecture_id,
            user_id=current_user.id,
            flashcard_index=review.flashcard_index
        )
        db.add(progress)
    
    ease_factor, interval, repetitions, next_review = calculate_next_review(
        review.quality,
        progress.ease_factor,
        progress.interval,
        progress.repetitions
    )
    
    progress.ease_factor = ease_factor
    progress.interval = interval
    progress.repetitions = repetitions
    progress.next_review = next_review
    progress.last_reviewed = datetime.utcnow()
    
    db.commit()
    db.refresh(progress)
    
    return progress

@app.get("/api/groups")
async def get_groups(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    groups = db.query(StudyGroup).filter(
        StudyGroup.members.contains([current_user.id])
    ).all()
    return groups

@app.post("/api/groups")
async def create_group(
    group: GroupCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    invite_code = secrets.token_urlsafe(8)
    
    new_group = StudyGroup(
        name=group.name,
        description=group.description,
        creator_id=current_user.id,
        invite_code=invite_code,
        members=[current_user.id]
    )
    
    db.add(new_group)
    db.commit()
    db.refresh(new_group)
    
    return new_group

@app.post("/api/groups/join/{invite_code}")
async def join_group(
    invite_code: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    group = db.query(StudyGroup).filter(StudyGroup.invite_code == invite_code).first()
    
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")
    
    if current_user.id not in group.members:
        group.members.append(current_user.id)
        db.commit()
        db.refresh(group)
    
    return group

@app.post("/api/groups/{group_id}/share/{lecture_id}")
async def share_lecture(
    group_id: int,
    lecture_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    group = db.query(StudyGroup).filter(StudyGroup.id == group_id).first()
    lecture = db.query(Lecture).filter(Lecture.id == lecture_id).first()
    
    if not group or not lecture:
        raise HTTPException(status_code=404, detail="Group or lecture not found")
    
    if current_user.id not in group.members:
        raise HTTPException(status_code=403, detail="Not a member")
    
    if lecture_id not in group.shared_lectures:
        group.shared_lectures.append(lecture_id)
        db.commit()
    
    return {"message": "Lecture shared successfully"}

@app.post("/api/export")
async def export_lecture(
    request: ExportRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    lecture = db.query(Lecture).filter(
        Lecture.id == request.lecture_id,
        Lecture.owner_id == current_user.id
    ).first()
    
    if not lecture:
        raise HTTPException(status_code=404, detail="Lecture not found")
    
    if request.format == "pdf":
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor='#1a1a1a',
            spaceAfter=30
        )
        story.append(Paragraph(lecture.title, title_style))
        story.append(Spacer(1, 12))
        
        # Summary
        story.append(Paragraph("Summary", styles['Heading2']))
        story.append(Paragraph(lecture.summary, styles['BodyText']))
        story.append(Spacer(1, 12))
        
        # Key Points
        story.append(Paragraph("Key Points", styles['Heading2']))
        for point in lecture.key_points:
            story.append(Paragraph(f"• {point}", styles['BodyText']))
        story.append(PageBreak())
        
        # Flashcards
        story.append(Paragraph("Flashcards", styles['Heading2']))
        for card in lecture.flashcards:
            story.append(Paragraph(f"<b>Q:</b> {card['front']}", styles['BodyText']))
            story.append(Paragraph(f"<b>A:</b> {card['back']}", styles['BodyText']))
            story.append(Spacer(1, 12))
        
        doc.build(story)
        buffer.seek(0)
        
        from fastapi.responses import StreamingResponse
        return StreamingResponse(
            buffer,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={lecture.title}.pdf"}
        )
    
    elif request.format == "markdown":
        md_content = f"# {lecture.title}\n\n"
        md_content += f"## Summary\n\n{lecture.summary}\n\n"
        md_content += "## Key Points\n\n"
        for point in lecture.key_points:
            md_content += f"- {point}\n"
        md_content += "\n## Flashcards\n\n"
        for card in lecture.flashcards:
            md_content += f"**Q:** {card['front']}\n\n"
            md_content += f"**A:** {card['back']}\n\n"
        
        from fastapi.responses import Response
        return Response(
            content=md_content,
            media_type="text/markdown",
            headers={"Content-Disposition": f"attachment; filename={lecture.title}.md"}
        )

@app.get("/")
async def root():
    return {
        "message": "Lecture Learning Platform API",
        "status": "running",
        "ai_provider": AI_PROVIDER,
        "features": {
            "video_transcription": True,
            "ai_summaries": True,
            "quizzes": True,
            "flashcards": True,
            "spaced_repetition": True,
            "study_groups": True,
            "export": ["pdf", "markdown"]
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)