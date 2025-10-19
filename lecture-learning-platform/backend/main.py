from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from typing import List, Optional
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import openai
import os
import json
import tempfile
import subprocess
from pathlib import Path
import uvicorn
from dotenv import load_dotenv
import secrets
import yt_dlp
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import io

# Import our modules
from database import get_db, init_db
from models import User, Lecture, LectureProgress, FlashcardProgress, StudyGroup, GroupMessage
from auth import create_access_token, get_current_user, ACCESS_TOKEN_EXPIRE_MINUTES

load_dotenv()

app = FastAPI(title="Lecture Learning Platform - Advanced")

# Initialize database
init_db()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

openai.api_key = os.getenv("OPENAI_API_KEY")

# ============== Pydantic Models ==============

class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class LectureCreate(BaseModel):
    title: str
    description: Optional[str] = None
    video_url: Optional[str] = None

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
    quality: int  # 0-5 rating for spaced repetition

class GroupCreate(BaseModel):
    name: str
    description: Optional[str] = None

class GroupMessage(BaseModel):
    group_id: int
    message: str

class ExportRequest(BaseModel):
    lecture_id: int
    format: str  # 'pdf' or 'markdown'

# ============== Helper Functions ==============

def extract_audio(video_path: str, audio_path: str):
    try:
        subprocess.run([
            'ffmpeg', '-i', video_path,
            '-vn', '-acodec', 'pcm_s16le',
            '-ar', '16000', '-ac', '1',
            audio_path
        ], check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError:
        return False

def download_video_from_url(url: str, output_path: str) -> str:
    """Download video from URL using yt-dlp"""
    ydl_opts = {
        'format': 'best[ext=mp4]/best',
        'outtmpl': output_path,
        'quiet': True,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return ydl.prepare_filename(info)

def transcribe_audio_with_timestamps(file_path: str) -> dict:
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
                "start": segment['start'],
                "end": segment['end'],
                "text": segment['text']
            })
            full_text.append(segment['text'])
        
        return {
            "full_text": " ".join(full_text),
            "segments": segments,
            "duration": transcript.duration if hasattr(transcript, 'duration') else segments[-1]['end'] if segments else 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

def get_transcript_by_timerange(segments: List[dict], start_min: float, end_min: Optional[float]) -> str:
    start_sec = start_min * 60
    end_sec = end_min * 60 if end_min else float('inf')
    
    filtered_segments = [
        seg['text'] for seg in segments
        if seg['start'] >= start_sec and seg['start'] <= end_sec
    ]
    
    return " ".join(filtered_segments)

def generate_summary(transcript: str, start_time: float = 0, end_time: Optional[float] = None) -> dict:
    time_context = ""
    if start_time > 0 or end_time:
        end_str = f"{end_time:.1f}" if end_time else "end"
        time_context = f"\n\nNote: This is a summary for the time range {start_time:.1f} to {end_str} minutes of the lecture."
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert educational content analyzer."},
                {"role": "user", "content": f"""Analyze this lecture transcript and provide:
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summary generation failed: {str(e)}")

def generate_quiz(transcript: str) -> List[dict]:
    try:
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quiz generation failed: {str(e)}")

def generate_study_plan(transcript: str, key_points: List[str]) -> dict:
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert educational planner."},
                {"role": "user", "content": f"""Create a 7-day study plan for this lecture content.

Key Points: {json.dumps(key_points)}

Respond in JSON format:
{{
    "overview": "Study plan overview",
    "days": [
        {{
            "day": 1,
            "focus": "Topic to focus on",
            "tasks": ["Task 1", "Task 2"],
            "duration": "30-45 minutes"
        }}
    ]
}}"""}
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Study plan generation failed: {str(e)}")

def generate_practice_problems(transcript: str) -> List[dict]:
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert at creating practice problems."},
                {"role": "user", "content": f"""Create 5 practice problems based on this lecture.

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
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Practice problems generation failed: {str(e)}")

def generate_flashcards(transcript: str, key_points: List[str]) -> List[dict]:
    """Generate flashcards from lecture content"""
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert at creating educational flashcards."},
                {"role": "user", "content": f"""Create 10 flashcards based on this lecture.

Key Points: {json.dumps(key_points)}
Transcript: {transcript[:3000]}

Create flashcards that test understanding of key concepts.

Respond in JSON format:
{{
    "flashcards": [
        {{
            "front": "Question or concept",
            "back": "Answer or explanation",
            "category": "topic category"
        }}
    ]
}}"""}
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)["flashcards"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Flashcard generation failed: {str(e)}")

def calculate_next_review(quality: int, ease_factor: float, interval: int, repetitions: int):
    """SM-2 Spaced Repetition Algorithm"""
    if quality < 3:
        # Failed - reset
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

def export_to_markdown(lecture: Lecture, db: Session) -> str:
    """Export lecture to Markdown format"""
    md = f"# {lecture.title}\n\n"
    
    if lecture.description:
        md += f"**Description:** {lecture.description}\n\n"
    
    md += f"**Duration:** {int(lecture.duration // 60)}:{int(lecture.duration % 60):02d}\n\n"
    md += "---\n\n"
    
    md += "## Summary\n\n"
    md += f"{lecture.summary}\n\n"
    
    md += "## Key Points\n\n"
    for i, point in enumerate(lecture.key_points, 1):
        md += f"{i}. {point}\n"
    md += "\n"
    
    if lecture.flashcards:
        md += "## Flashcards\n\n"
        for i, card in enumerate(lecture.flashcards, 1):
            md += f"### Card {i}\n"
            md += f"**Q:** {card['front']}\n\n"
            md += f"**A:** {card['back']}\n\n"
    
    md += "## Study Plan\n\n"
    for day in lecture.study_plan.get('days', []):
        md += f"### Day {day['day']}: {day['focus']}\n"
        md += f"*Duration: {day['duration']}*\n\n"
        for task in day['tasks']:
            md += f"- {task}\n"
        md += "\n"
    
    md += "## Full Transcript\n\n"
    md += lecture.transcript
    
    return md

def export_to_pdf(lecture: Lecture, db: Session) -> bytes:
    """Export lecture to PDF format"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=24, spaceAfter=30)
    story.append(Paragraph(lecture.title, title_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Summary
    story.append(Paragraph("Summary", styles['Heading2']))
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph(lecture.summary, styles['BodyText']))
    story.append(Spacer(1, 0.3*inch))
    
    # Key Points
    story.append(Paragraph("Key Points", styles['Heading2']))
    story.append(Spacer(1, 0.1*inch))
    for i, point in enumerate(lecture.key_points, 1):
        story.append(Paragraph(f"{i}. {point}", styles['BodyText']))
        story.append(Spacer(1, 0.1*inch))
    
    story.append(PageBreak())
    
    # Flashcards
    if lecture.flashcards:
        story.append(Paragraph("Flashcards", styles['Heading2']))
        story.append(Spacer(1, 0.1*inch))
        for i, card in enumerate(lecture.flashcards, 1):
            story.append(Paragraph(f"<b>Q{i}:</b> {card['front']}", styles['BodyText']))
            story.append(Paragraph(f"<b>A:</b> {card['back']}", styles['BodyText']))
            story.append(Spacer(1, 0.2*inch))
    
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

# ============== Authentication Endpoints ==============

@app.post("/api/auth/register")
async def register(user: UserCreate, db: Session = Depends(get_db)):
    # Check if user exists
    if db.query(User).filter(User.username == user.username).first():
        raise HTTPException(status_code=400, detail="Username already exists")
    if db.query(User).filter(User.email == user.email).first():
        raise HTTPException(status_code=400, detail="Email already exists")
    
    # Create new user
    new_user = User(username=user.username, email=user.email)
    new_user.set_password(user.password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    # Create token
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
async def get_me(current_user: User = Depends(lambda token, db: get_current_user(token, db)), db: Session = Depends(get_db)):
    return {
        "id": current_user.id,
        "username": current_user.username,
        "email": current_user.email
    }

# ============== Lecture Endpoints ==============

@app.post("/api/upload-lecture")
async def upload_lecture(
    file: Optional[UploadFile] = File(None),
    title: str = Form(...),
    description: Optional[str] = Form(None),
    video_url: Optional[str] = Form(None),
    current_user: User = Depends(lambda token, db: get_current_user(token, db)),
    db: Session = Depends(get_db)
):
    video_path = None
    audio_path = None
    
    try:
        # Determine video source
        if file:
            # Upload file
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
                content = await file.read()
                tmp_file.write(content)
                video_path = tmp_file.name
            video_source = "upload"
            source_url = None
        elif video_url:
            # Download from URL
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                video_path = tmp_file.name
            video_path = download_video_from_url(video_url, video_path)
            video_source = "url"
            source_url = video_url
        else:
            raise HTTPException(status_code=400, detail="Either file or video_url must be provided")
        
        # Extract audio
        audio_path = video_path.replace(Path(video_path).suffix, '.wav')
        extract_audio(video_path, audio_path)
        
        transcribe_path = audio_path if os.path.exists(audio_path) else video_path
        
        # Transcribe
        transcript_data = transcribe_audio_with_timestamps(transcribe_path)
        
        # Generate content
        summary_data = generate_summary(transcript_data['full_text'])
        quiz = generate_quiz(transcript_data['full_text'])
        study_plan = generate_study_plan(transcript_data['full_text'], summary_data['key_points'])
        practice_problems = generate_practice_problems(transcript_data['full_text'])
        flashcards = generate_flashcards(transcript_data['full_text'], summary_data['key_points'])
        
        # Create lecture in database
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
        
        return lecture
        
    finally:
        # Cleanup
        if video_path and os.path.exists(video_path):
            os.remove(video_path)
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)

@app.get("/api/lectures")
async def get_lectures(
    current_user: User = Depends(lambda token, db: get_current_user(token, db)),
    db: Session = Depends(get_db)
):
    lectures = db.query(Lecture).filter(Lecture.owner_id == current_user.id).all()
    return lectures

@app.get("/api/lectures/{lecture_id}")
async def get_lecture(
    lecture_id: int,
    current_user: User = Depends(lambda token, db: get_current_user(token, db)),
    db: Session = Depends(get_db)
):
    lecture = db.query(Lecture).filter(Lecture.id == lecture_id).first()
    if not lecture:
        raise HTTPException(status_code=404, detail="Lecture not found")
    
    # Check if user has access
    if lecture.owner_id != current_user.id:
        # Check if shared via group
        shared = any(group in current_user.groups for group in lecture.shared_with_groups)
        if not shared:
            raise HTTPException(status_code=403, detail="Access denied")
    
    return lecture

@app.post("/api/timestamp-summary")
async def get_timestamp_summary(
    request: TimestampSummaryRequest,
    current_user: User = Depends(lambda token, db: get_current_user(token, db)),
    db: Session = Depends(get_db)
):
    lecture = db.query(Lecture).filter(Lecture.id == request.lecture_id).first()
    if not lecture:
        raise HTTPException(status_code=404, detail="Lecture not found")
    
    transcript_segment = get_transcript_by_timerange(
        lecture.transcript_with_timestamps,
        request.start_time,
        request.end_time
    )
    
    if not transcript_segment:
        raise HTTPException(status_code=400, detail="No content in specified time range")
    
    summary_data = generate_summary(transcript_segment, request.start_time, request.end_time)
    
    return {
        "time_range": {
            "start": request.start_time,
            "end": request.end_time or lecture.duration / 60
        },
        "summary": summary_data['summary'],
        "key_points": summary_data['key_points'],
        "transcript": transcript_segment
    }

# ============== Progress Tracking ==============

@app.post("/api/progress")
async def update_progress(
    progress: ProgressUpdate,
    current_user: User = Depends(lambda token, db: get_current_user(token, db)),
    db: Session = Depends(get_db)
):
    prog = db.query(LectureProgress).filter(
        LectureProgress.user_id == current_user.id,
        LectureProgress.lecture_id == progress.lecture_id
    ).first()
    
    if not prog:
        prog = LectureProgress(user_id=current_user.id, lecture_id=progress.lecture_id)
        db.add(prog)
    
    if progress.last_position is not None:
        prog.last_position = progress.last_position
    if progress.completed is not None:
        prog.completed = progress.completed
    if progress.notes is not None:
        prog.notes = progress.notes
    
    db.commit()
    db.refresh(prog)
    return prog

@app.get("/api/progress/{lecture_id}")
async def get_progress(
    lecture_id: int,
    current_user: User = Depends(lambda token, db: get_current_user(token, db)),
    db: Session = Depends(get_db)
):
    prog = db.query(LectureProgress).filter(
        LectureProgress.user_id == current_user.id,
        LectureProgress.lecture_id == lecture_id
    ).first()
    
    if not prog:
        return {"message": "No progress tracked yet"}
    
    return prog

# ============== Quiz ==============

@app.post("/api/submit-quiz")
async def submit_quiz(
    quiz_data: QuizSubmit,
    current_user: User = Depends(lambda token, db: get_current_user(token, db)),
    db: Session = Depends(get_db)
):
    lecture = db.query(Lecture).filter(Lecture.id == quiz_data.lecture_id).first()
    if not lecture:
        raise HTTPException(status_code=404, detail="Lecture not found")
    
    quiz = lecture.quiz
    results = []
    correct_count = 0
    
    for i, answer in enumerate(quiz_data.answers):
        is_correct = answer == quiz[i]['correct_answer']
        if is_correct:
            correct_count += 1
        
        results.append({
            "question_index": i,
            "correct": is_correct,
            "user_answer": answer,
            "correct_answer": quiz[i]['correct_answer'],
            "explanation": quiz[i]['explanation']
        })
    
    # Update progress
    prog = db.query(LectureProgress).filter(
        LectureProgress.user_id == current_user.id,
        LectureProgress.lecture_id == quiz_data.lecture_id
    ).first()
    
    if not prog:
        prog = LectureProgress(user_id=current_user.id, lecture_id=quiz_data.lecture_id)
        db.add(prog)
    
    prog.quiz_taken = True
    prog.quiz_score = correct_count
    prog.quiz_attempts += 1
    db.commit()
    
    return {
        "score": correct_count,
        "total": len(quiz),
        "percentage": (correct_count / len(quiz)) * 100,
        "results": results
    }

# ============== Flashcards & Spaced Repetition ==============

@app.get("/api/flashcards/{lecture_id}")
async def get_flashcards(
    lecture_id: int,
    current_user: User = Depends(lambda token, db: get_current_user(token, db)),
    db: Session = Depends(get_db)
):
    lecture = db.query(Lecture).filter(Lecture.id == lecture_id).first()
    if not lecture:
        raise HTTPException(status_code=404, detail="Lecture not found")
    
    return lecture.flashcards

@app.get("/api/flashcards/due/{lecture_id}")
async def get_due_flashcards(
    lecture_id: int,
    current_user: User = Depends(lambda token, db: get_current_user(token, db)),
    db: Session = Depends(get_db)
):
    """Get flashcards due for review"""
    lecture = db.query(Lecture).filter(Lecture.id == lecture_id).first()
    if not lecture:
        raise HTTPException(status_code=404, detail="Lecture not found")
    
    # Get all flashcard progress for this lecture
    progress_records = db.query(FlashcardProgress).filter(
        FlashcardProgress.user_id == current_user.id,
        FlashcardProgress.lecture_id == lecture_id,
        FlashcardProgress.next_review <= datetime.utcnow()
    ).all()
    
    # If no progress, return first 5 cards
    if not progress_records:
        return lecture.flashcards[:5]
    
    due_cards = []
    for prog in progress_records:
        if prog.flashcard_index < len(lecture.flashcards):
            card = lecture.flashcards[prog.flashcard_index].copy()
            card['progress'] = {
                'ease_factor': prog.ease_factor,
                'interval': prog.interval,
                'times_seen': prog.times_seen,
                'accuracy': prog.times_correct / prog.times_seen if prog.times_seen > 0 else 0
            }
            due_cards.append(card)
    
    return due_cards

@app.post("/api/flashcards/review")
async def review_flashcard(
    review: FlashcardReview,
    current_user: User = Depends(lambda token, db: get_current_user(token, db)),
    db: Session = Depends(get_db)
):
    """Submit flashcard review with quality rating"""
    prog = db.query(FlashcardProgress).filter(
        FlashcardProgress.user_id == current_user.id,
        FlashcardProgress.lecture_id == review.lecture_id,
        FlashcardProgress.flashcard_index == review.flashcard_index
    ).first()
    
    if not prog:
        prog = FlashcardProgress(
            user_id=current_user.id,
            lecture_id=review.lecture_id,
            flashcard_index=review.flashcard_index
        )
        db.add(prog)
    
    # Update stats
    prog.times_seen += 1
    if review.quality >= 3:
        prog.times_correct += 1
    
    prog.last_reviewed = datetime.utcnow()
    
    # Calculate next review using SM-2 algorithm
    ease, interval, reps, next_rev = calculate_next_review(
        review.quality,
        prog.ease_factor,
        prog.interval,
        prog.repetitions
    )
    
    prog.ease_factor = ease
    prog.interval = interval
    prog.repetitions = reps
    prog.next_review = next_rev
    
    db.commit()
    db.refresh(prog)
    
    return {
        "next_review": next_rev,
        "interval_days": interval,
        "message": "Review recorded successfully"
    }

# ============== Study Groups ==============

@app.post("/api/groups")
async def create_group(
    group: GroupCreate,
    current_user: User = Depends(lambda token, db: get_current_user(token, db)),
    db: Session = Depends(get_db)
):
    invite_code = secrets.token_urlsafe(8)
    new_group = StudyGroup(
        name=group.name,
        description=group.description,
        invite_code=invite_code,
        creator_id=current_user.id
    )
    new_group.members.append(current_user)
    
    db.add(new_group)
    db.commit()
    db.refresh(new_group)
    
    return new_group

@app.post("/api/groups/join/{invite_code}")
async def join_group(
    invite_code: str,
    current_user: User = Depends(lambda token, db: get_current_user(token, db)),
    db: Session = Depends(get_db)
):
    group = db.query(StudyGroup).filter(StudyGroup.invite_code == invite_code).first()
    if not group:
        raise HTTPException(status_code=404, detail="Invalid invite code")
    
    if current_user not in group.members:
        group.members.append(current_user)
        db.commit()
    
    return group

@app.get("/api/groups")
async def get_groups(
    current_user: User = Depends(lambda token, db: get_current_user(token, db)),
    db: Session = Depends(get_db)
):
    return current_user.groups

@app.post("/api/groups/{group_id}/share/{lecture_id}")
async def share_lecture_with_group(
    group_id: int,
    lecture_id: int,
    current_user: User = Depends(lambda token, db: get_current_user(token, db)),
    db: Session = Depends(get_db)
):
    lecture = db.query(Lecture).filter(Lecture.id == lecture_id, Lecture.owner_id == current_user.id).first()
    if not lecture:
        raise HTTPException(status_code=404, detail="Lecture not found or not owned by you")
    
    group = db.query(StudyGroup).filter(StudyGroup.id == group_id).first()
    if not group or current_user not in group.members:
        raise HTTPException(status_code=403, detail="Access denied")
    
    if group not in lecture.shared_with_groups:
        lecture.shared_with_groups.append(group)
        db.commit()
    
    return {"message": "Lecture shared with group"}

# ============== Export ==============

@app.post("/api/export")
async def export_lecture(
    request: ExportRequest,
    current_user: User = Depends(lambda token, db: get_current_user(token, db)),
    db: Session = Depends(get_db)
):
    from fastapi.responses import Response
    
    lecture = db.query(Lecture).filter(Lecture.id == request.lecture_id).first()
    if not lecture:
        raise HTTPException(status_code=404, detail="Lecture not found")
    
    if request.format == "markdown":
        content = export_to_markdown(lecture, db)
        return Response(
            content=content,
            media_type="text/markdown",
            headers={"Content-Disposition": f"attachment; filename={lecture.title}.md"}
        )
    elif request.format == "pdf":
        content = export_to_pdf(lecture, db)
        return Response(
            content=content,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={lecture.title}.pdf"}
        )
    else:
        raise HTTPException(status_code=400, detail="Invalid format")

@app.get("/")
async def root():
    return {"message": "Lecture Learning Platform API - Advanced", "status": "running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)