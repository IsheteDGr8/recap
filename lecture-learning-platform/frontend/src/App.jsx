import React, { useState, useEffect } from 'react';
import './App.css';

const API_URL = 'http://localhost:8000';

function App() {
  // Auth state
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [token, setToken] = useState(localStorage.getItem('token'));
  const [user, setUser] = useState(null);
  const [authMode, setAuthMode] = useState('login');

  // App state
  const [lectures, setLectures] = useState([]);
  const [selectedLecture, setSelectedLecture] = useState(null);
  const [activeTab, setActiveTab] = useState('summary');
  const [uploading, setUploading] = useState(false);
  
  // Quiz state
  const [quizAnswers, setQuizAnswers] = useState([]);
  const [quizResults, setQuizResults] = useState(null);
  
  // Timestamp state
  const [startTime, setStartTime] = useState(0);
  const [endTime, setEndTime] = useState('');
  const [timestampSummary, setTimestampSummary] = useState(null);
  const [loadingTimestamp, setLoadingTimestamp] = useState(false);
  
  // Flashcard state
  const [flashcards, setFlashcards] = useState([]);
  const [currentCardIndex, setCurrentCardIndex] = useState(0);
  const [showAnswer, setShowAnswer] = useState(false);
  const [dueCards, setDueCards] = useState([]);
  
  // Study group state
  const [groups, setGroups] = useState([]);
  const [selectedGroup, setSelectedGroup] = useState(null);
  const [showGroupModal, setShowGroupModal] = useState(false);
  
  // Progress state
  const [progress, setProgress] = useState({});
  const [notes, setNotes] = useState('');
  
  // Upload state
  const [uploadType, setUploadType] = useState('file');
  const [videoUrl, setVideoUrl] = useState('');
  const [lectureTitle, setLectureTitle] = useState('');
  const [lectureDescription, setLectureDescription] = useState('');

  useEffect(() => {
    if (token) {
      validateToken();
    }
  }, []);

  useEffect(() => {
    if (isAuthenticated) {
      fetchLectures();
      fetchGroups();
    }
  }, [isAuthenticated]);

  useEffect(() => {
    if (selectedLecture && isAuthenticated) {
      fetchProgress();
      loadFlashcards();
    }
  }, [selectedLecture]);

  const apiCall = async (url, options = {}) => {
    const headers = {
      ...options.headers,
    };
    
    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
    }
    
    if (!(options.body instanceof FormData)) {
      headers['Content-Type'] = 'application/json';
    }
    
    const response = await fetch(`${API_URL}${url}`, {
      ...options,
      headers,
    });
    
    if (response.status === 401) {
      logout();
      throw new Error('Unauthorized');
    }
    
    return response;
  };

  const validateToken = async () => {
    try {
      const response = await apiCall('/api/auth/me');
      if (response.ok) {
        const userData = await response.json();
        setUser(userData);
        setIsAuthenticated(true);
      } else {
        logout();
      }
    } catch (error) {
      logout();
    }
  };

  const login = async (username, password) => {
    try {
      const formData = new URLSearchParams();
      formData.append('username', username);
      formData.append('password', password);
      
      const response = await fetch(`${API_URL}/api/auth/login`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: formData,
      });
      
      if (response.ok) {
        const data = await response.json();
        setToken(data.access_token);
        localStorage.setItem('token', data.access_token);
        setUser({ username: data.username, email: data.email });
        setIsAuthenticated(true);
      } else {
        alert('Invalid credentials');
      }
    } catch (error) {
      console.error('Login error:', error);
      alert('Login failed');
    }
  };

  const register = async (username, email, password) => {
    try {
      const response = await fetch(`${API_URL}/api/auth/register`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ username, email, password }),
      });
      
      if (response.ok) {
        const data = await response.json();
        const newToken = data.access_token;
        setToken(newToken);
        localStorage.setItem('token', newToken);
        setUser({ username: data.username, email: data.email });
        setIsAuthenticated(true);
        // Give token state time to update
        setTimeout(() => {
          fetchLectures();
        }, 100);
      } else {
        const error = await response.json();
        alert(error.detail || 'Registration failed');
      }
    } catch (error) {
      console.error('Registration error:', error);
      alert('Registration failed');
    }
  };

  const logout = () => {
    setToken(null);
    localStorage.removeItem('token');
    setIsAuthenticated(false);
    setUser(null);
    setLectures([]);
    setSelectedLecture(null);
  };

  const fetchLectures = async () => {
    try {
      const response = await apiCall('/api/lectures');
      if (!response.ok) {
        console.error('Failed to fetch lectures:', response.status);
        setLectures([]);
        return;
      }
      const data = await response.json();
      // Ensure data is an array
      setLectures(Array.isArray(data) ? data : []);
    } catch (error) {
      console.error('Error fetching lectures:', error);
      setLectures([]);
    }
  };

  const fetchProgress = async () => {
    try {
      const response = await apiCall(`/api/progress/${selectedLecture.id}`);
      const data = await response.json();
      setProgress(data);
      if (data.notes) setNotes(data.notes);
    } catch (error) {
      console.error('Error fetching progress:', error);
    }
  };

  const saveProgress = async (updates) => {
    try {
      await apiCall('/api/progress', {
        method: 'POST',
        body: JSON.stringify({
          lecture_id: selectedLecture.id,
          ...updates
        }),
      });
    } catch (error) {
      console.error('Error saving progress:', error);
    }
  };

  const handleFileUpload = async (event) => {
    event.preventDefault();
    
    if (uploadType === 'file' && !event.target.file?.files[0]) {
      alert('Please select a file');
      return;
    }
    
    if (uploadType === 'url' && !videoUrl) {
      alert('Please enter a video URL');
      return;
    }
    
    if (!lectureTitle) {
      alert('Please enter a lecture title');
      return;
    }

    setUploading(true);
    const formData = new FormData();
    
    if (uploadType === 'file') {
      formData.append('file', event.target.file.files[0]);
    } else {
      formData.append('video_url', videoUrl);
    }
    
    formData.append('title', lectureTitle);
    if (lectureDescription) {
      formData.append('description', lectureDescription);
    }

    try {
      const response = await apiCall('/api/upload-lecture', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        setLectures([...lectures, data]);
        setSelectedLecture(data);
        setLectureTitle('');
        setLectureDescription('');
        setVideoUrl('');
        alert('Lecture uploaded and processed successfully!');
      }
    } catch (error) {
      console.error('Error uploading lecture:', error);
      alert('Error uploading lecture');
    } finally {
      setUploading(false);
    }
  };

  const handleTimestampSummary = async () => {
    if (!selectedLecture) return;

    setLoadingTimestamp(true);
    try {
      const response = await apiCall('/api/timestamp-summary', {
        method: 'POST',
        body: JSON.stringify({
          lecture_id: selectedLecture.id,
          start_time: parseFloat(startTime) || 0,
          end_time: endTime ? parseFloat(endTime) : null,
        }),
      });

      const data = await response.json();
      setTimestampSummary(data);
    } catch (error) {
      console.error('Error generating timestamp summary:', error);
      alert('Error generating summary');
    } finally {
      setLoadingTimestamp(false);
    }
  };

  const formatDuration = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const handleQuizSubmit = async () => {
    if (quizAnswers.length !== selectedLecture.quiz.length) {
      alert('Please answer all questions');
      return;
    }

    try {
      const response = await apiCall('/api/submit-quiz', {
        method: 'POST',
        body: JSON.stringify({
          lecture_id: selectedLecture.id,
          answers: quizAnswers,
        }),
      });

      const results = await response.json();
      setQuizResults(results);
    } catch (error) {
      console.error('Error submitting quiz:', error);
    }
  };

  const resetQuiz = () => {
    setQuizAnswers([]);
    setQuizResults(null);
  };

  // Flashcard functions
  const loadFlashcards = async () => {
    try {
      const response = await apiCall(`/api/flashcards/${selectedLecture.id}`);
      const data = await response.json();
      setFlashcards(data);
      
      // Load due cards
      const dueResponse = await apiCall(`/api/flashcards/due/${selectedLecture.id}`);
      const dueData = await dueResponse.json();
      setDueCards(dueData);
    } catch (error) {
      console.error('Error loading flashcards:', error);
    }
  };

  const reviewFlashcard = async (quality) => {
    try {
      await apiCall('/api/flashcards/review', {
        method: 'POST',
        body: JSON.stringify({
          lecture_id: selectedLecture.id,
          flashcard_index: currentCardIndex,
          quality: quality
        }),
      });
      
      // Move to next card
      if (currentCardIndex < flashcards.length - 1) {
        setCurrentCardIndex(currentCardIndex + 1);
        setShowAnswer(false);
      } else {
        alert('All flashcards reviewed!');
        loadFlashcards();
      }
    } catch (error) {
      console.error('Error reviewing flashcard:', error);
    }
  };

  // Group functions
  const fetchGroups = async () => {
    try {
      const response = await apiCall('/api/groups');
      const data = await response.json();
      setGroups(data);
    } catch (error) {
      console.error('Error fetching groups:', error);
    }
  };

  const createGroup = async (name, description) => {
    try {
      const response = await apiCall('/api/groups', {
        method: 'POST',
        body: JSON.stringify({ name, description }),
      });
      
      if (response.ok) {
        const data = await response.json();
        setGroups([...groups, data]);
        setShowGroupModal(false);
        alert(`Group created! Invite code: ${data.invite_code}`);
      }
    } catch (error) {
      console.error('Error creating group:', error);
    }
  };

  const joinGroup = async (inviteCode) => {
    try {
      const response = await apiCall(`/api/groups/join/${inviteCode}`, {
        method: 'POST',
      });
      
      if (response.ok) {
        fetchGroups();
        alert('Joined group successfully!');
      }
    } catch (error) {
      console.error('Error joining group:', error);
      alert('Invalid invite code');
    }
  };

  const shareWithGroup = async (groupId) => {
    if (!selectedLecture) return;
    
    try {
      await apiCall(`/api/groups/${groupId}/share/${selectedLecture.id}`, {
        method: 'POST',
      });
      alert('Lecture shared with group!');
    } catch (error) {
      console.error('Error sharing lecture:', error);
    }
  };

  const exportLecture = async (format) => {
    if (!selectedLecture) return;
    
    try {
      const response = await apiCall('/api/export', {
        method: 'POST',
        body: JSON.stringify({
          lecture_id: selectedLecture.id,
          format: format
        }),
      });
      
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${selectedLecture.title}.${format === 'pdf' ? 'pdf' : 'md'}`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (error) {
      console.error('Error exporting lecture:', error);
    }
  };

  // Auth form
  if (!isAuthenticated) {
    return (
      <div className="auth-container">
        <div className="auth-box">
          <h1>📚 Lecture Learning Platform</h1>
          <p>Advanced learning with AI-powered study tools</p>
          
          <div className="auth-tabs">
            <button
              className={authMode === 'login' ? 'active' : ''}
              onClick={() => setAuthMode('login')}
            >
              Login
            </button>
            <button
              className={authMode === 'register' ? 'active' : ''}
              onClick={() => setAuthMode('register')}
            >
              Register
            </button>
          </div>
          
          <form onSubmit={(e) => {
            e.preventDefault();
            const formData = new FormData(e.target);
            if (authMode === 'login') {
              login(formData.get('username'), formData.get('password'));
            } else {
              register(
                formData.get('username'),
                formData.get('email'),
                formData.get('password')
              );
            }
          }}>
            <input
              type="text"
              name="username"
              placeholder="Username"
              required
            />
            {authMode === 'register' && (
              <input
                type="email"
                name="email"
                placeholder="Email"
                required
              />
            )}
            <input
              type="password"
              name="password"
              placeholder="Password"
              required
            />
            <button type="submit">
              {authMode === 'login' ? 'Login' : 'Register'}
            </button>
          </form>
        </div>
      </div>
    );
  }

  return (
    <div className="App">
      <header className="header">
        <div>
          <h1>📚 Lecture Learning Platform</h1>
          <p>Welcome, {user?.username}!</p>
        </div>
        <div className="header-actions">
          <button onClick={() => setShowGroupModal(true)} className="group-btn">
            👥 Groups ({groups.length})
          </button>
          <button onClick={logout} className="logout-btn">Logout</button>
        </div>
      </header>

      <div className="container">
        <div className="sidebar">
          <div className="upload-section">
            <h2>Upload Lecture</h2>
            
            <div className="upload-type-selector">
              <button
                className={uploadType === 'file' ? 'active' : ''}
                onClick={() => setUploadType('file')}
              >
                📁 File Upload
              </button>
              <button
                className={uploadType === 'url' ? 'active' : ''}
                onClick={() => setUploadType('url')}
              >
                🔗 Video URL
              </button>
            </div>
            
            <form onSubmit={handleFileUpload}>
              <input
                type="text"
                placeholder="Lecture Title *"
                value={lectureTitle}
                onChange={(e) => setLectureTitle(e.target.value)}
                required
              />
              
              <textarea
                placeholder="Description (optional)"
                value={lectureDescription}
                onChange={(e) => setLectureDescription(e.target.value)}
                rows="2"
              />
              
              {uploadType === 'file' ? (
                <input
                  type="file"
                  name="file"
                  accept="video/*"
                  disabled={uploading}
                />
              ) : (
                <input
                  type="url"
                  placeholder="YouTube/Video URL"
                  value={videoUrl}
                  onChange={(e) => setVideoUrl(e.target.value)}
                  required
                />
              )}
              
              <button type="submit" disabled={uploading} className="upload-button">
                {uploading ? '⏳ Processing...' : '📤 Upload & Process'}
              </button>
            </form>
            {uploading && <p className="upload-status">This may take several minutes...</p>}
          </div>

          <div className="lectures-list">
            <h2>My Lectures</h2>
            {lectures.length === 0 ? (
              <p className="empty-state">No lectures yet</p>
            ) : (
              lectures.map((lecture) => (
                <div
                  key={lecture.id}
                  className={`lecture-item ${selectedLecture?.id === lecture.id ? 'active' : ''}`}
                  onClick={() => {
                    setSelectedLecture(lecture);
                    setActiveTab('summary');
                    resetQuiz();
                  }}
                >
                  <div className="lecture-icon">
                    {lecture.video_source === 'url' ? '🔗' : '🎥'}
                  </div>
                  <div className="lecture-info">
                    <div className="lecture-title">{lecture.title}</div>
                    <div className="lecture-meta">
                      {formatDuration(lecture.duration)}
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        <div className="main-content">
          {!selectedLecture ? (
            <div className="welcome-screen">
              <h2>Welcome! 🎓</h2>
              <p>Upload a lecture to get started with AI-powered learning</p>
              <ul>
                <li>✨ Upload files or paste video URLs</li>
                <li>📝 Timestamped summaries</li>
                <li>🃏 Smart flashcards with spaced repetition</li>
                <li>👥 Study groups & collaboration</li>
                <li>📊 Progress tracking</li>
                <li>📄 Export to PDF/Markdown</li>
              </ul>
            </div>
          ) : (
            <>
              <div className="lecture-header">
                <div>
                  <h2>{selectedLecture.title}</h2>
                  {selectedLecture.description && <p>{selectedLecture.description}</p>}
                </div>
                <div className="lecture-actions">
                  <button onClick={() => exportLecture('markdown')} className="export-btn">
                    📝 Export MD
                  </button>
                  <button onClick={() => exportLecture('pdf')} className="export-btn">
                    📄 Export PDF
                  </button>
                </div>
              </div>

              <div className="tabs">
                <button className={`tab ${activeTab === 'summary' ? 'active' : ''}`} onClick={() => setActiveTab('summary')}>📋 Summary</button>
                <button className={`tab ${activeTab === 'timestamp' ? 'active' : ''}`} onClick={() => setActiveTab('timestamp')}>⏱️ Time Range</button>
                <button className={`tab ${activeTab === 'flashcards' ? 'active' : ''}`} onClick={() => setActiveTab('flashcards')}>🃏 Flashcards</button>
                <button className={`tab ${activeTab === 'quiz' ? 'active' : ''}`} onClick={() => setActiveTab('quiz')}>❓ Quiz</button>
                <button className={`tab ${activeTab === 'study-plan' ? 'active' : ''}`} onClick={() => setActiveTab('study-plan')}>📅 Study Plan</button>
                <button className={`tab ${activeTab === 'practice' ? 'active' : ''}`} onClick={() => setActiveTab('practice')}>💪 Practice</button>
                <button className={`tab ${activeTab === 'progress' ? 'active' : ''}`} onClick={() => setActiveTab('progress')}>📈 Progress</button>
              </div>

              <div className="content">
                {activeTab === 'summary' && (
                  <div className="summary-section">
                    <h2>Full Lecture Summary</h2>
                    <div className="summary-content">
                      <p>{selectedLecture.summary}</p>
                    </div>

                    <h3>Key Points</h3>
                    <div className="key-points">
                      {selectedLecture.key_points.map((point, index) => (
                        <div key={index} className="key-point">
                          <span className="point-number">{index + 1}</span>
                          <span className="point-text">{point}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {activeTab === 'timestamp' && (
                  <div className="timestamp-section">
                    <h2>Time Range Summary</h2>
                    <div className="time-range-selector">
                      <div className="time-input-group">
                        <label>
                          Start (min):
                          <input type="number" value={startTime} onChange={(e) => setStartTime(e.target.value)} />
                        </label>
                        <label>
                          End (min):
                          <input type="number" value={endTime} onChange={(e) => setEndTime(e.target.value)} placeholder="End of lecture" />
                        </label>
                      </div>
                      <button onClick={handleTimestampSummary} disabled={loadingTimestamp} className="generate-button">
                        {loadingTimestamp ? '⏳ Generating...' : '🎯 Generate Summary'}
                      </button>
                    </div>

                    {timestampSummary && (
                      <div className="timestamp-results">
                        <h3>Summary</h3>
                        <p>{timestampSummary.summary}</p>
                        <h4>Key Points</h4>
                        <div className="key-points">
                          {timestampSummary.key_points.map((point, i) => (
                            <div key={i} className="key-point">
                              <span className="point-number">{i + 1}</span>
                              <span className="point-text">{point}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {activeTab === 'flashcards' && (
                  <div className="flashcards-section">
                    <h2>Flashcards ({flashcards.length} total, {dueCards.length} due)</h2>
                    
                    {flashcards.length > 0 && currentCardIndex < flashcards.length && (
                      <div className="flashcard-container">
                        <div className={`flashcard ${showAnswer ? 'flipped' : ''}`}>
                          <div className="flashcard-inner">
                            <div className="flashcard-front">
                              <h3>Question</h3>
                              <p>{flashcards[currentCardIndex].front}</p>
                              <button onClick={() => setShowAnswer(true)}>Show Answer</button>
                            </div>
                            <div className="flashcard-back">
                              <h3>Answer</h3>
                              <p>{flashcards[currentCardIndex].back}</p>
                              <div className="quality-buttons">
                                <p>How well did you know this?</p>
                                <button onClick={() => reviewFlashcard(1)} className="quality-btn fail">Again (1)</button>
                                <button onClick={() => reviewFlashcard(3)} className="quality-btn hard">Hard (3)</button>
                                <button onClick={() => reviewFlashcard(4)} className="quality-btn good">Good (4)</button>
                                <button onClick={() => reviewFlashcard(5)} className="quality-btn easy">Easy (5)</button>
                              </div>
                            </div>
                          </div>
                        </div>
                        <p className="card-progress">Card {currentCardIndex + 1} of {flashcards.length}</p>
                      </div>
                    )}
                  </div>
                )}

                {activeTab === 'quiz' && (
                  <div className="quiz-section">
                    <h2>Knowledge Check Quiz</h2>
                    {!quizResults ? (
                      <>
                        {selectedLecture.quiz.map((question, qIndex) => (
                          <div key={qIndex} className="quiz-question">
                            <h3>Question {qIndex + 1}</h3>
                            <p className="question-text">{question.question}</p>
                            <div className="options">
                              {question.options.map((option, oIndex) => (
                                <label key={oIndex} className="option">
                                  <input
                                    type="radio"
                                    name={`question-${qIndex}`}
                                    checked={quizAnswers[qIndex] === oIndex}
                                    onChange={() => {
                                      const newAnswers = [...quizAnswers];
                                      newAnswers[qIndex] = oIndex;
                                      setQuizAnswers(newAnswers);
                                    }}
                                  />
                                  <span>{option}</span>
                                </label>
                              ))}
                            </div>
                          </div>
                        ))}
                        <button className="submit-button" onClick={handleQuizSubmit} disabled={quizAnswers.length !== selectedLecture.quiz.length}>
                          Submit Quiz
                        </button>
                      </>
                    ) : (
                      <div className="quiz-results">
                        <div className="score-card">
                          <h3>Your Score</h3>
                          <div className="score">{quizResults.score} / {quizResults.total}</div>
                          <div className="percentage">{quizResults.percentage.toFixed(0)}%</div>
                        </div>
                        {quizResults.results.map((result, index) => (
                          <div key={index} className={`result-item ${result.correct ? 'correct' : 'incorrect'}`}>
                            <h4>Question {index + 1}</h4>
                            <p className="question-text">{selectedLecture.quiz[index].question}</p>
                            <p className="result-status">{result.correct ? '✅ Correct!' : '❌ Incorrect'}</p>
                            {!result.correct && (
                              <p className="correct-answer">Correct answer: {selectedLecture.quiz[index].options[result.correct_answer]}</p>
                            )}
                            <div className="explanation">
                              <strong>Explanation:</strong> {result.explanation}
                            </div>
                          </div>
                        ))}
                        <button className="retry-button" onClick={resetQuiz}>Try Again</button>
                      </div>
                    )}
                  </div>
                )}

                {activeTab === 'study-plan' && (
                  <div className="study-plan-section">
                    <h2>7-Day Study Plan</h2>
                    <p className="plan-overview">{selectedLecture.study_plan.overview}</p>
                    <div className="days-grid">
                      {selectedLecture.study_plan.days.map((day) => (
                        <div key={day.day} className="day-card">
                          <div className="day-header">
                            <h3>Day {day.day}</h3>
                            <span className="duration">{day.duration}</span>
                          </div>
                          <h4 className="focus">{day.focus}</h4>
                          <ul className="tasks">
                            {day.tasks.map((task, index) => (
                              <li key={index}>{task}</li>
                            ))}
                          </ul>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {activeTab === 'practice' && (
                  <div className="practice-section">
                    <h2>Practice Problems</h2>
                    {selectedLecture.practice_problems.map((problem, index) => (
                      <div key={index} className="practice-problem">
                        <div className="problem-header">
                          <h3>Problem {index + 1}</h3>
                          <span className={`difficulty ${problem.difficulty}`}>{problem.difficulty}</span>
                        </div>
                        <p className="problem-text">{problem.problem}</p>
                        <details className="hint-details">
                          <summary>💡 Show Hint</summary>
                          <p>{problem.hint}</p>
                        </details>
                        <details className="solution-details">
                          <summary>✅ Show Solution</summary>
                          <p>{problem.solution}</p>
                        </details>
                      </div>
                    ))}
                  </div>
                )}

                {activeTab === 'progress' && (
                  <div className="progress-section">
                    <h2>Your Progress</h2>
                    <div className="progress-stats">
                      <div className="stat-card">
                        <h3>Completed</h3>
                        <p className="stat-value">{progress.completed ? 'Yes ✅' : 'No'}</p>
                      </div>
                      <div className="stat-card">
                        <h3>Quiz Score</h3>
                        <p className="stat-value">{progress.quiz_score !== undefined ? `${progress.quiz_score}/${selectedLecture.quiz.length}` : 'Not taken'}</p>
                      </div>
                      <div className="stat-card">
                        <h3>Quiz Attempts</h3>
                        <p className="stat-value">{progress.quiz_attempts || 0}</p>
                      </div>
                    </div>
                    
                    <div className="notes-section">
                      <h3>My Notes</h3>
                      <textarea
                        value={notes}
                        onChange={(e) => setNotes(e.target.value)}
                        placeholder="Add your notes here..."
                        rows="10"
                      />
                      <button onClick={() => saveProgress({ notes })} className="save-notes-btn">
                        Save Notes
                      </button>
                    </div>
                    
                    <div className="progress-actions">
                      <button onClick={() => saveProgress({ completed: true })} className="complete-btn">
                        Mark as Complete
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </>
          )}
        </div>
      </div>

      {showGroupModal && (
        <div className="modal-overlay" onClick={() => setShowGroupModal(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <h2>Study Groups</h2>
            <button className="modal-close" onClick={() => setShowGroupModal(false)}>×</button>
            
            <div className="group-actions">
              <form onSubmit={(e) => {
                e.preventDefault();
                const formData = new FormData(e.target);
                createGroup(formData.get('name'), formData.get('description'));
              }}>
                <h3>Create Group</h3>
                <input type="text" name="name" placeholder="Group Name" required />
                <textarea name="description" placeholder="Description" rows="2" />
                <button type="submit">Create</button>
              </form>
              
              <form onSubmit={(e) => {
                e.preventDefault();
                const formData = new FormData(e.target);
                joinGroup(formData.get('code'));
                e.target.reset();
              }}>
                <h3>Join Group</h3>
                <input type="text" name="code" placeholder="Invite Code" required />
                <button type="submit">Join</button>
              </form>
            </div>
            
            <h3>My Groups</h3>
            <div className="groups-list">
              {groups.map(group => (
                <div key={group.id} className="group-item">
                  <h4>{group.name}</h4>
                  <p>Code: {group.invite_code}</p>
                  {selectedLecture && (
                    <button onClick={() => shareWithGroup(group.id)} className="share-btn">
                      Share Current Lecture
                    </button>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;