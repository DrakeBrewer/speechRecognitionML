Recording instructions:

Live Demo Audio Collection Instructions

Goal: Collect ~10 minutes of varied speech per speaker for training a robust speaker recognition model.


1. Setup

Use the same laptop mic (or the mic you’ll use for the live demo).

Use the provided record() Python function to ensure consistent sample rate (22050 Hz) and WAV format.

Record in a quiet environment to minimize background noise. The script can be ran from terminal in the project root dir using the following cmd: 

python3 -m src.speech_recognition.utils.record_audio --speaker <yourname> --session <sessionID> --duration <duration>. 

For example -> python3 -m src.speech_recognition.utils.record_audio --speaker David --session 01 --duration 180. 


2. Recording Sessions

Each speaker should record roughly 10 minutes total, split into 2–4 separate sessions of variable lengths.

Example 1: 2 × 2-minute + 2 × 3-minute sessions

Example 2: 3 × 3-minute + 1 × 1-minute session

File naming convention: Python record_audio script will save file as "speaker_sessionID.wav"


3. Content Suggestions

Vary the content so the model doesn’t just memorize words:

Read aloud paragraphs from books (different from other team members).

Random sentences (news articles, Wikipedia, jokes, short stories).

Poems, lyrics, or quotes.

Spontaneous speech (talk about your day, hobbies, or tell a short story).

Try to mix long and short sentences to give a variety of prosody and pitch.


4. Recording Tips

Keep speaking volume consistent. Avoid whispering or yelling.

Avoid excessive background noise.

If a session has long silence periods, it’s fine; the feature extractor will skip silent frames automatically.

Don’t worry about exact durations — just aim for roughly the total target time per speaker.
