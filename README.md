# Meeting Summarizer

## Overview
The **Meeting Summarizer** is a Python project that transcribes meeting audio files and generates concise, action-oriented summaries.  
It uses **OpenAI Whisper** for transcription and a **free online Groq LLM** for summarization, extracting:

- **Key Points** – Main discussion topics  
- **Key Decisions** – Important outcomes and agreements  
- **Action Items** – Tasks assigned to team members  

All transcripts and summaries can optionally be stored locally in **SQLite**.

> **Note:** The repository includes a **sample audio file** (`example_audio/sample_meeting.mp3`) for demonstration purposes.

---

## Features
- Transcribe audio files (MP3, WAV, M4A) using Whisper  
- Summarize transcripts using Groq LLM  
- Generate structured summaries with key points, decisions, and tasks  
- Store transcripts and summaries in SQLite  
- Easy-to-use command line interface  

---

## Installation

1. Clone the repository:

```bash
git clone <your-repo-url>
cd meeting-summarizer
