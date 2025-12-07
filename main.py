# -------------------------------
# 🌐 FastAPI Sarkari Sahayak Backend (Context-Aware, Structured Output)
# -------------------------------
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import FileResponse
from openai import OpenAI
import uvicorn
import json
import requests
from google.cloud import vision # Updated to use Google Vision API
from vosk import Model, KaldiRecognizer
import wave
# NOTE: Removed pytesseract and related imports and configuration.
# The following imports are retained but no longer used in `analyze_form`
# They might be relevant if image manipulation were added later.
from datetime import datetime

# -------------------------------
# 🤖 Initialize OpenAI Client
# -------------------------------
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -------------------------------
# 🌟 Initialize Google Vision Client
# -------------------------------
# NOTE: This assumes 'google-cloud-vision' is installed and 
# GOOGLE_APPLICATION_CREDENTIALS is set up in the environment.
try:
    vision_client = vision.ImageAnnotatorClient()
    VISION_ENABLED = True
except Exception as e:
    # If the environment lacks the correct setup or library
    vision_client = None
    VISION_ENABLED = False
    print(f"WARNING: Google Vision client failed to initialize: {e}")

# -------------------------------
# 🚀 Initialize FastAPI App
# -------------------------------
app = FastAPI(title="Sarkari Sahayak API", version="5.0")

# ✅ Allow Frontend Access (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://sarkari-sahayek-frontend-1.onrender.com",
    "http://localhost:5173", ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# 📥 Define Request Model
# -------------------------------
class ChatRequest(BaseModel):
    message: str
    language: str
    session_id: str  # track conversation context per user

# -------------------------------
# 🧠 In-memory session store
# -------------------------------
sessions = {}

# -------------------------------
# 🧠 Sarkari Sahayak Chat Endpoint
# -------------------------------
@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    if req.session_id not in sessions:
        sessions[req.session_id] = []

    session_history = sessions[req.session_id]
    session_history.append({"role": "user", "content": req.message})

    system_prompt = f"""
You are **Sarkari Sahayak**, a highly knowledgeable AI assistant that explains 
government schemes, legal rights, and civic duties.

Instructions for output:
1. Always reply in JSON format only.
2. Structure:
{{
  "answer": "<Main explanation in {req.language}>",
  "steps": ["Step 1", "Step 2", ...],
  "tables": [{{"headers": ["H1", "H2"], "rows": [["R1C1", "R1C2"], ["R2C1","R2C2"]]}}],
  "links": [
    {{"label": "Official website", "url": "https://www.example.gov"}},
    {{"label": "YouTube tutorial", "url": "https://www.youtube.com/watch?v=VIDEOID"}}
  ]
}}
3. Include step-by-step instructions if relevant.
4. Include real links only; do not make up URLs.
5. Include tables for structured data if needed.
6. Use bullets, numbered lists, and formatting naturally.
7. Output directly in the user's preferred language ("{req.language}").
8. Maintain context from previous messages (you can reference prior Q&A from the session).
9. Keep tone polite, helpful, trustworthy.
"""

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system_prompt}] + session_history,
            temperature=0.7,
            max_tokens=1000,
        )

        raw_reply = completion.choices[0].message.content.strip()
        session_history.append({"role": "assistant", "content": raw_reply})

        try:
            data = json.loads(raw_reply)
            return data
        except json.JSONDecodeError:
            # If the LLM returns non-JSON text unexpectedly
            return {"answer": raw_reply, "steps": [], "tables": [], "links": []}

    except Exception as e:
        print("Error:", str(e))
        return {"answer": "⚠️ Server error. Please try again later.", "steps": [], "tables": [], "links": []}

# -------------------------------
# 🧾 Document & Form Helper (Now using Google Vision)
# -------------------------------
@app.post("/api/analyze_form")
async def analyze_form(file: UploadFile = File(...), session_id: str = Form(default="")):
    """
    1️⃣ Accepts image or screenshot of a government form.
    2️⃣ Detects empty or blank fields via Google Vision OCR.
    3️⃣ Uses LLM to suggest what should be filled.
    4️⃣ Returns only structured JSON (no images).
    """
    if not VISION_ENABLED or vision_client is None:
        return {"answer": "Google Vision API is not configured on the server. OCR functionality is disabled.", "steps": [], "tables": [], "links": []}
    
    try:
        content = await file.read()
        extracted_text = ""

        # Use Google Vision API for text detection (Full Document Text Detection)
        try:
            # The client handles the API call and image processing from raw bytes
            image = vision.Image(content=content)
            
            # Use DOCUMENT_TEXT_DETECTION for detailed, full-page OCR
            response = vision_client.document_text_detection(image=image)
            extracted_text = response.full_text_annotation.text
            
        except Exception as e:
            print(f"Google Vision API Error: {e}")
            return {"answer": "OCR failed. Could not process image using Google Vision.", "steps": [], "tables": [], "links": []}

        # LLM prompt for missing fields
        prompt = f"""
You are an expert in Indian government forms.
Below is text extracted from a user's uploaded form:
---
{extracted_text}
---
Some fields are blank or have empty boxes.
Identify what information should be filled in each blank field.
Respond strictly in JSON:
{{
  "answer": "Summary of the form type",
  "steps": ["Field 1 - what to fill", "Field 2 - what to fill", ...],
  "tables": [],
  "links": []
}}
"""

        try:
            llm = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You analyze and assist with Indian government forms."},
                    {"role": "user", "content": prompt}
                ]
            )
            suggestions = llm.choices[0].message.content.strip()
            try:
                parsed = json.loads(suggestions)
            except:
                # Fallback if LLM doesn't return perfect JSON
                parsed = {"answer": suggestions, "steps": [], "tables": [], "links": []}
        except Exception as e:
            parsed = {"answer": f"AI analysis failed: {e}", "steps": [], "tables": [], "links": []}

        # Return structured JSON
        return parsed

    except Exception as e:
        print("Error analyzing form:", e)
        return {"answer": f"Error analyzing form: {e}", "steps": [], "tables": [], "links": []}

# ⚡ Frontend alias to prevent 404
@app.post("/api/upload_document")
async def upload_document(file: UploadFile = File(...), session_id: str = Form(default="")):
    return await analyze_form(file, session_id=session_id)

# -------------------------------
# 📰 Notifications System
# -------------------------------
all_notifications = []

def fetch_notifications():
    url = "https://apisetu.gov.in/public/sector/notifications"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                global all_notifications
                new_items = [n for n in data if n not in all_notifications]
                all_notifications = new_items + all_notifications
                return data
            else:
                return all_notifications[:3]
        else:
            return all_notifications[:3]
    except Exception as e:
        print("Error fetching notifications:", e)
        return all_notifications[:3]

@app.get("/api/notifications")
async def get_notifications():
    return fetch_notifications()
# -------------------------------
# 📋 Eligibility Model
# -------------------------------
class EligibilityRequest(BaseModel):
    state: str
    caste: str
    gender: str
    occupation: str

# -------------------------------
# 🏛️ Eligibility Endpoint
# -------------------------------
@app.post("/api/eligibility")
async def check_eligibility(req: EligibilityRequest):
    """
    Takes user's demographic and occupation info and returns
    a list of government schemes the person is eligible for.
    Response is strictly JSON with:
    {
        "answer": "<summary>",
        "schemes": [
            {"name": "Scheme 1", "link": "https://...", "description": "..."},
            ...
        ]
    }
    """
    prompt = f"""
You are a government schemes expert who gives tailored, *real* scheme suggestions based on user's profile.

User details:
- State: {req.state}
- Caste: {req.caste}
- Gender: {req.gender}
- Occupation: {req.occupation}

Respond only in JSON and include *only relevant schemes*.

Output format:
{{
  "answer": "Brief summary about eligibility in one sentence",
  "schemes": [
    {{
      "name": "Scheme name",
      "link": "https://official.gov.in/real-url",
      "description": "One-line real description of the scheme and why user qualifies"
    }}
  ]
}}

Rules:
1. Use real Indian government schemes and their official links (central or state).
2. Consider user’s caste, gender, and occupation for eligibility.
3. Prioritize **state-specific schemes** if the state is mentioned.
4. Always return at least one result; if uncertain, include a general central scheme.
5. Respond *only* in valid JSON.
6. Avoid repeating the same set for all users.
Examples:
- A woman entrepreneur → Stand Up India, Mahila E-Haat.
- A farmer → PM-Kisan, KCC, Fasal Bima Yojana.
- SC/ST student → Post-Matric Scholarship Scheme, NSP portal.
- Unemployed youth → PM Kaushal Vikas Yojana, state employment scheme.
"""


    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant providing government scheme eligibility."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,  # lower temp for more deterministic output
            max_tokens=1000,
        )
        raw_reply = completion.choices[0].message.content.strip()

        try:
            data = json.loads(raw_reply)
            return data
        except json.JSONDecodeError:
            # If LLM still returns non-JSON, attempt to extract JSON from text
            import re
            match = re.search(r"\{.*\}", raw_reply, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except:
                    pass
            return {
                "answer": raw_reply,
                "schemes": []
            }

    except Exception as e:
        print("Eligibility check error:", e)
        return {
            "answer": "⚠️ Could not fetch eligible schemes. Please try again later.",
            "schemes": []
        }MODEL_DIR = "/tmp/vosk-model-small-en-us-0.15"
MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
VOSK_MODEL = None

def load_vosk_model():
    global VOSK_MODEL

    try:
        if not os.path.exists(MODEL_DIR):
            print("⏳ Downloading Vosk model...")
            zip_path = "/tmp/model.zip"
            r = requests.get(MODEL_URL)
            with open(zip_path, "wb") as f:
                f.write(r.content)

            print("📦 Extracting model...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall("/tmp")

        print("🚀 Loading Vosk model...")
        VOSK_MODEL = Model(MODEL_DIR)
        print("✔ Vosk model ready")

    except Exception as e:
        print(f"❌ Model load error: {e}")
        VOSK_MODEL = None

# Load once when server starts
load_vosk_model()

# -------------------------------
#   Voice to Text API
# -------------------------------
@app.post("/api/voice_to_text")
async def voice_to_text_endpoint(file: UploadFile = File(...)):
    """
    1️⃣ Accepts audio upload
    2️⃣ Transcribes using offline Vosk engine
    3️⃣ Returns clean text
    """

    if not VOSK_MODEL:
        return {"text": "Error: Vosk model not loaded on server.", "error": True}

    try:
        audio_content = await file.read()

        # Save to /tmp
        temp_filename = f"/tmp/{file.filename}_{datetime.now().timestamp()}.wav"
        with open(temp_filename, "wb") as f:
            f.write(audio_content)

        # Vosk supports only PCM 16kHz mono WAV
        wf = wave.open(temp_filename, "rb")
        if wf.getnchannels() != 1 or wf.getframerate() != 16000:
            wf.close()
            os.remove(temp_filename)
            return {
                "text": "Audio must be 16kHz mono WAV. Convert before upload.",
                "error": True
            }

        rec = KaldiRecognizer(VOSK_MODEL, wf.getframerate())
        rec.SetWords(False)

        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            rec.AcceptWaveform(data)

        result = rec.FinalResult()
        text = json.loads(result).get("text", "")

        wf.close()
        os.remove(temp_filename)

        return {"text": text, "error": False}

    except Exception as e:
        if 'temp_filename' in locals() and os.path.exists(temp_filename):
            os.remove(temp_filename)
        return {"text": f"Error during transcription: {str(e)}", "error": True}
# -------------------------------
# ▶️ Run Server
# -------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
