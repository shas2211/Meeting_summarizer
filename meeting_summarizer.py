import argparse
import sqlite3
import os
import sys
import datetime
import textwrap

try:
    import whisper
except Exception as e:
    whisper = None

try:
    from transformers import pipeline
except Exception as e:
    pipeline = None

DB_PATH = 'meetings.db'


def init_db(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS meetings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT,
            transcript TEXT,
            summary TEXT,
            action_items TEXT,
            created_at TEXT
        )
    ''')
    conn.commit()
    conn.close()


def transcribe_whisper(audio_path, model_name='small'):
    if whisper is None:
        raise RuntimeError("whisper package is not installed. Run: pip install -U openai-whisper")
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    print(f"Loading Whisper model '{model_name}' (this may take a while)...")
    model = whisper.load_model(model_name)
    print("Transcribing... This can take time depending on model & hardware.")
    result = model.transcribe(audio_path)
    transcript = result.get('text', '').strip()
    print("Transcription complete. Transcript length:", len(transcript))
    return transcript


def get_summarizer(model_name='sshleifer/distilbart-cnn-12-6'):
    if pipeline is None:
        raise RuntimeError("transformers package not installed. Run: pip install transformers")
    # Use summarization pipeline
    print(f"Loading summarization model '{model_name}' (this may take a while)...")
    summarizer = pipeline('summarization', model=model_name)
    return summarizer


def summarize_transcript(transcript, summarizer, max_length=200, min_length=30):
    # If transcript is long, split into chunks
    CHUNK_TOKENS = 800  # approximate token chunk size for summarizer
    paragraphs = textwrap.wrap(transcript, CHUNK_TOKENS)
    summaries = []
    for p in paragraphs:
        out = summarizer(p, max_length=max_length, min_length=min_length, truncation=True)
        summaries.append(out[0]['summary_text'].strip())
    # Join partial summaries and compress again
    combined = '\n'.join(summaries)
    if len(combined) > CHUNK_TOKENS:
        final = summarizer(combined, max_length=max_length, min_length=min_length, truncation=True)[0]['summary_text'].strip()
        return final
    return combined


def extract_action_items(transcript, summarizer, max_length=150, min_length=10):
    # Attempt to extract action items by instructing the summarizer
    prompt = (
        "Extract action items from the meeting transcript below.\n"
        "Output as bullet points. If none, output 'No clear action items found.'\n\n"
        f"Transcript:\n{transcript}"
    )
    # For models that don't accept long prompts, chunk transcript to the last ~2000 chars
    if len(prompt) > 4000:
        prompt = prompt[-4000:]
    out = summarizer(prompt, max_length=max_length, min_length=min_length, truncation=True)
    items = out[0]['summary_text'].strip()
    return items


def save_meeting(file_name, transcript, summary, action_items, db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute('''
        INSERT INTO meetings (file_name, transcript, summary, action_items, created_at)
        VALUES (?, ?, ?, ?, ?)
    ''', (file_name, transcript, summary, action_items, datetime.datetime.utcnow().isoformat()))
    conn.commit()
    meeting_id = cur.lastrowid
    conn.close()
    return meeting_id


def view_meeting(meeting_id, db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute('SELECT id, file_name, transcript, summary, action_items, created_at FROM meetings WHERE id=?', (meeting_id,))
    row = cur.fetchone()
    conn.close()
    return row


def list_meetings(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute('SELECT id, file_name, created_at FROM meetings ORDER BY created_at DESC')
    rows = cur.fetchall()
    conn.close()
    return rows


def export_meeting(meeting_id, out_path, db_path=DB_PATH):
    row = view_meeting(meeting_id, db_path)
    if not row:
        raise ValueError('Meeting not found')
    id_, file_name, transcript, summary, actions, created_at = row
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(f"# Meeting {id_}\n")
        f.write(f"File: {file_name}\n")
        f.write(f"Created at (UTC): {created_at}\n\n")
        f.write("## Summary\n\n")
        f.write(summary + "\n\n")
        f.write("## Action Items\n\n")
        f.write(actions + "\n\n")
        f.write("## Transcript\n\n")
        f.write(transcript + "\n")
    return out_path

from groq import Groq

def summarize_meeting_text(transcript_text, api_key):
    """
    Uses Groq's free LLM API to summarize meeting transcript into
    key points, key decisions, and action items.
    """
    client = Groq(api_key=api_key)

    prompt = f"""
    You are an AI meeting summarizer.
    Summarize the following meeting transcript into:
    1. **Key Points**
    2. **Key Decisions**
    3. **Action Items**
    
    Make the output clear and bullet-pointed.

    Transcript:
    {transcript_text}
    """

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",

        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    return completion.choices[0].message.content


def main():
    parser = argparse.ArgumentParser(description='Meeting Summarizer CLI (transcribe + summarize + store)')
    sub = parser.add_subparsers(dest='cmd')

    p_transcribe = sub.add_parser('transcribe')
    p_transcribe.add_argument('--file', required=True, help='Path to audio file')
    p_transcribe.add_argument('--model', default='small', help='Whisper model name (tiny, base, small, medium, large)')

    p_sum_store = sub.add_parser('summarize_store')
    p_sum_store.add_argument('--file', required=True, help='Path to audio file')
    p_sum_store.add_argument('--whisper_model', default='small', help='Whisper model name')
    p_sum_store.add_argument('--summ_model', default='sshleifer/distilbart-cnn-12-6', help='Summarization model name')

    p_view = sub.add_parser('view')
    p_view.add_argument('--id', type=int, required=True, help='Meeting ID')

    p_list = sub.add_parser('list')

    p_export = sub.add_parser('export')
    p_export.add_argument('--id', type=int, required=True, help='Meeting ID')
    p_export.add_argument('--out', required=True, help='Output markdown path')

    args = parser.parse_args()

    init_db()

    if args.cmd == 'transcribe':
        transcript = transcribe_whisper(args.file, model_name=args.model)
        print('\n----- TRANSCRIPT -----\n')
        print(transcript)
        print("\nGenerating meeting summary using Groq LLM...\n")

        api_key = "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"  # paste your key here
        summary = summarize_meeting_text(transcript, api_key)

        print(summary)
        return

    if args.cmd == 'summarize_store':
        transcript = transcribe_whisper(args.file, model_name=args.whisper_model)
        summarizer = get_summarizer(model_name=args.summ_model)
        print('Generating summary...')
        summary = summarize_transcript(transcript, summarizer)
        print('Extracting action items...')
        action_items = extract_action_items(transcript, summarizer)
        meeting_id = save_meeting(os.path.basename(args.file), transcript, summary, action_items)
        print(f"Saved meeting with id={meeting_id}")
        print('\n--- SUMMARY ---\n')
        print(summary)
        print('\n--- ACTION ITEMS ---\n')
        print(action_items)
        return

    if args.cmd == 'view':
        row = view_meeting(args.id)
        if not row:
            print('Meeting not found')
            return
        id_, file_name, transcript, summary, actions, created_at = row
        print(f"ID: {id_}\nFile: {file_name}\nCreated: {created_at}\n\nSUMMARY:\n{summary}\n\nACTIONS:\n{actions}\n\nTRANSCRIPT:\n{transcript}")
        return

    if args.cmd == 'list':
        rows = list_meetings()
        if not rows:
            print('No meetings found')
            return
        for r in rows:
            print(f"{r[0]} | {r[1]} | {r[2]}")
        return

    if args.cmd == 'export':
        path = export_meeting(args.id, args.out)
        print(f'Exported to {path}')
        return

    parser.print_help()


if __name__ == '__main__':
    main()
