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
import pytesseract
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"


from PIL import Image, ImageDraw
import numpy as np
import io
import cv2
from datetime import datetime

# -------------------------------
# 🤖 Initialize OpenAI Client
# -------------------------------
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
            return {"answer": raw_reply, "steps": [], "tables": [], "links": []}

    except Exception as e:
        print("Error:", str(e))
        return {"answer": "⚠️ Server error. Please try again later.", "steps": [], "tables": [], "links": []}

# -------------------------------
# 🧾 Document & Form Helper
# -------------------------------
@app.post("/api/analyze_form")
async def analyze_form(file: UploadFile = File(...), session_id: str = Form(default="")):
    """
    1️⃣ Accepts image or screenshot of a government form.
    2️⃣ Detects empty or blank fields via OCR.
    3️⃣ Uses LLM to suggest what should be filled.
    4️⃣ Returns only structured JSON (no images).
    """
    try:
        content = await file.read()

        # Convert to image for OCR
        img_cv = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
        if img_cv is None:
            return {"answer": "Invalid image format", "steps": [], "tables": [], "links": []}

        # OCR extract text
        extracted_text = pytesseract.image_to_string(img_cv)

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
                parsed = {"answer": suggestions, "steps": [], "tables": [], "links": []}
        except Exception as e:
            parsed = {"answer": f"AI analysis failed: {e}", "steps": [], "tables": [], "links": []}

        # Return only structured JSON (no images)
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
        }

# -------------------------------
# ▶️ Run Server
# -------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
