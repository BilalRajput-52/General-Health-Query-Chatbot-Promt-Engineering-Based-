# MediGuide — General Health Query Chatbot

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-API-orange?style=for-the-badge&logo=groq&logoColor=white)
![LLaMA](https://img.shields.io/badge/LLaMA-3.3--70B-purple?style=for-the-badge&logo=meta&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-Groq-green?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge)

# Tasks Overview

| Task | Title | Status |
|------|-------|--------|
| Task 1 | Exploring and Visualizing the Iris Dataset |  Complete |
| Task 2 | Predict Future Stock Prices |  complete |
| Task 3 | Heart Disease Prediction |  complete |
| Task 4 | General Health Query Chatbot |  complete |
| Task 5 | Mental Health Support Chatbot |  Upcoming |
| Task 6 | House Price Prediction |  Upcoming |


**Task 4 — AI/ML Engineering Internship**
**DevelopersHub Corporation | Due: 28 April, 2026**

*A prompt-engineered health information chatbot powered by LLaMA 3.3 70B via Groq API*

</div>

---

##  Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Architecture](#-project-architecture)
- [Prompt Engineering](#-prompt-engineering)
- [Safety System](#-safety-system)
- [Installation & Setup](#-installation--setup)
- [How to Run](#-how-to-run)
- [Example Queries & Outputs](#-example-queries--outputs)
- [Multi-turn Conversation](#-multi-turn-conversation)
- [Prompt Engineering Analysis](#-prompt-engineering-analysis)
- [Project Structure](#-project-structure)
- [Key Results & Findings](#-key-results--findings)
- [Future Improvements](#-future-improvements)
- [Disclaimer](#-disclaimer)

---

##  Overview

**MediGuide** is an AI-powered **General Health Query Chatbot** built as Task 4 of the DevelopersHub Corporation AI/ML Engineering Internship. It leverages **Meta's LLaMA 3.3 70B** large language model served through the **Groq API** (world's fastest LLM inference engine) to answer general health-related questions in a safe, friendly, and informative manner.

The project demonstrates real-world application of:
- **Prompt Engineering** — crafting system prompts that control LLM behavior
- **Safety Filtering** — multi-layer protection to prevent harmful outputs
- **Multi-turn Conversation** — maintaining context across multiple user messages
- **LLM API Integration** — using ChatGroq with LangChain for production-ready code

> ⚠️ **Important:** MediGuide is designed strictly for **general health education**. It does not provide medical diagnosis, treatment plans, or prescription recommendations.

---

##  Problem Statement

**Objective:** Create a chatbot that can answer general health-related questions using a Large Language Model (LLM), with prompt engineering to ensure:
- Responses are friendly and clearly written
- Safety filters prevent harmful or dangerous medical advice
- The chatbot redirects users to professional healthcare providers when needed

**Source:** DevelopersHub Corporation AI/ML Engineering Internship — Task 4

---

##  Features

| Feature | Description |
|---------|-------------|
|  **LLaMA 3.3 70B** | State-of-the-art open-source model with 70 billion parameters |
|  **Groq API** | Ultra-fast inference — up to 800 tokens/second |
|  **Prompt Engineering** | 6 advanced techniques applied to system prompt design |
|  **Two-Layer Safety** | Python pre-filter + LLM-level guardrails |
|  **Emergency Detection** | Auto-detects emergencies and redirects to 115/1122 |
|  **Multi-turn Memory** | Remembers full conversation context |
|  **Secure API Key** | `.env` file pattern — no hardcoded credentials |
|  **Interactive CLI** | Live chat interface in the terminal |
|  **Prompt Comparison** | Demonstrates V1 vs V2 vs V3 prompt effectiveness |

---

##  Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Language** | Python 3.10+ | Core programming language |
| **LLM Model** | `llama-3.3-70b-versatile` | The AI brain of the chatbot |
| **LLM Provider** | Groq API | Ultra-fast model inference |
| **LLM Framework** | LangChain + ChatGroq | Structured LLM calls & message history |
| **Environment** | python-dotenv | Secure API key management |
| **Notebook** | Jupyter Notebook | Development & demonstration |
| **Safety** | Python `re` (Regex) | Pre-processing safety filter |

### Why These Choices?

**Groq + LLaMA 3.3 70B vs Alternatives:**

| Criteria | Groq + LLaMA 3.3 70B | OpenAI GPT-3.5 | Mistral-7B |
|----------|----------------------|----------------|------------|
| Speed |  ~800 tok/sec | ~60 tok/sec | Varies |
| Cost |  Free tier |  Paid | Free |
| Model Size | 70B parameters | ~175B | 7B |
| Quality | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| Open Source |  Yes (Meta) |  No |  Yes |
| Health Q&A | Excellent | Good | Limited |

---

##  Project Architecture

```
User Input
    │
    ▼
┌─────────────────────────────────┐
│     LAYER 1: Safety Filter      │  ← Python pre-processing (regex)
│  ┌─────────┐  ┌──────────────┐  │
│  │Emergency│  │ High-Risk    │  │
│  │Keywords │  │ Keywords     │  │
│  └────┬────┘  └──────┬───────┘  │
│       │              │           │
│  Instant      Add caution     │
│  Alert          context flag    │
└───────┼──────────────┼──────────┘
        │              │
        ▼              ▼
┌─────────────────────────────────┐
│     LAYER 2: LLaMA 3.3 70B      │  ← LLM via Groq API (ChatGroq)
│                                 │
│  System Prompt (Engineered) →   │
│  • Role: MediGuide persona      │
│  • Constraints: NEVER diagnose  │
│  • Format: bullet points        │
│  • Safety: doctor reminders     │
│  • Scope: education only        │
└─────────────────┬───────────────┘
                  │
                  ▼
         Safe, Helpful Response
                  │
                  ▼
        Conversation History
         (Multi-turn Memory)
```

---

##  Prompt Engineering

Prompt Engineering is the practice of crafting instructions that guide an LLM to produce desired outputs consistently and safely. This is the **core skill** demonstrated in Task 4.

### The System Prompt

The system prompt acts as the "constitution" of the chatbot — it defines who MediGuide is, how it communicates, and what rules it must always follow.

### 6 Techniques Applied

#### 1.  Role Prompting
Giving the model a clear identity and persona.
```
"You are MediGuide, a friendly and knowledgeable general health
information assistant created for EDUCATIONAL PURPOSES ONLY."
```
**Effect:** The model adopts a consistent persona, tone, and set of values throughout the conversation.

---

#### 2. Constraint Injection
Hard rules the model must strictly follow.
```
"NEVER diagnose a specific condition for the user"
"NEVER recommend specific prescription medications"
"NEVER encourage self-medication with prescription drugs"
```
**Effect:** The LLM reliably refuses harmful requests instead of trying to be "helpful" in dangerous ways.

---

#### 3.  Format Control
Specifying exactly how responses should be structured.
```
"Use short clear paragraphs for explanations"
"Use bullet points (•) when listing symptoms, tips, or causes"
"Keep responses under 250 words unless more detail is truly needed"
"End health responses with: ⚕️ Please consult a healthcare professional..."
```
**Effect:** Consistent, readable, professional output every time.

---

#### 4.  Tone Setting
Defining the emotional register and communication style.
```
"Be warm, empathetic, clear, and easy to understand"
"Use simple language; avoid overly technical jargon unless asked"
"Show genuine care for the user"
```
**Effect:** Responses feel human, approachable, and trustworthy — not robotic or clinical.

---

#### 5.  Scope Definition
Explicitly listing what the chatbot can and cannot help with.
```
## WHAT YOU CAN HELP WITH:
- General health education
- OTC medication general safety information
- Healthy lifestyle tips

## WHAT YOU MUST REFUSE:
- "Do I have [disease]?" → Cannot diagnose
- "Should I take [prescription drug]?" → Redirect to doctor
```
**Effect:** Eliminates ambiguity — the model knows exactly where its boundaries are.

---

#### 6.  Safety Guardrails
Emergency-specific instructions baked into the prompt.
```
"For EMERGENCIES (chest pain, difficulty breathing, stroke, severe bleeding)
→ IMMEDIATELY direct to: 115 or 1122 (Pakistan) or local emergency number"

"For mental health crises → provide Umang helpline: 0317-4288665 (Pakistan)"
```
**Effect:** Even if the Python safety filter misses something, the LLM itself knows to redirect emergencies.

---

##  Safety System

MediGuide uses a **two-layer safety architecture** to ensure no harmful advice is given.

### Layer 1 — Python Pre-processing Filter

Runs **before** the LLM is called. Uses regex pattern matching to instantly detect dangerous queries.

```python
# Emergency patterns → bypass LLM, return instant alert
EMERGENCY_PATTERNS = [
    r'chest\s*pain', r"can'?t\s*breathe",
    r'\bstroke\b', r'heart\s*attack',
    r'unconscious', r'severe\s*bleeding',
    r'suicide', r'overdose'
]

# High-risk patterns → still send to LLM but add extra caution
HIGH_RISK_PATTERNS = [
    r'diagnose\s*me', r'do\s*i\s*have',
    r'should\s*i\s*take', r'lethal\s*dose'
]
```

**Three possible outcomes:**

```
Emergency Detected  →  Bypass LLM entirely → Show 115/1122 alert
High-Risk Detected  →  Add caution note to LLM context → Proceed carefully  
Safe Query          →  Normal LLM call → Standard response
```

### Layer 2 — LLM-Level Guardrails (System Prompt)

Even if a query passes Layer 1, the system prompt instructs LLaMA 3.3 70B to:
- Refuse diagnosis requests
- Refuse prescription drug advice
- Always end with a "consult a doctor" reminder
- Handle edge cases with professional caution

### Safety Test Results

| Query | Layer 1 | Layer 2 | Final Response |
|-------|---------|---------|----------------|
| "I have chest pain" |  BLOCKED | Bypassed | Emergency 115/1122 alert |
| "Do I have diabetes?" |  FLAGGED | Extra caution | Refuses, redirects to doctor |
| "Is paracetamol safe?" | SAFE | Normal | General safety info + disclaimer |
| "What causes headaches?" |  SAFE | Normal | Informative educational response |

---

## 💻 Installation & Setup

### Prerequisites
- Python 3.10 or higher
- pip (Python package manager)
- A free Groq API key

### Step 1 — Clone the Repository
```bash
git clone https://github.com/BilalRajput-52/developershub-aiml-internship.git
cd developershub-aiml-internship/task4-health-chatbot
```

### Step 2 — Install Dependencies
```bash
pip install groq langchain-groq langchain-core python-dotenv
```

Or install from requirements file:
```bash
pip install -r requirements.txt
```

### Step 3 — Get Your Free Groq API Key
1. Visit  **https://console.groq.com**
2. Create a free account
3. Navigate to **API Keys** → **Create API Key**
4. Copy the key (starts with `gsk_...`)

### Step 4 — Configure Environment
Create a `.env` file in the project root:
```bash
# .env
GROQ_API_KEY=gsk_your_actual_api_key_here
```

> ⚠️ **Never commit your `.env` file to GitHub!** It is already in `.gitignore`.

### Step 5 — Verify Setup
```bash
python -c "from groq import Groq; print(' Setup successful!')"
```

---

## 🚀 How to Run

### Option 1 — Jupyter Notebook (Recommended)
```bash
jupyter notebook Task4_Health_Query_Chatbot.ipynb
```
Then run all cells top to bottom (`Cell → Run All`).

### Option 2 — Interactive CLI Chat
In the notebook, navigate to **Section 7** and uncomment:
```python
run_interactive_chat()
```
Then run that cell. Type your questions live in the terminal. Type `quit` to exit.

### Option 3 — Quick Python Script
```python
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))

response = llm.invoke([
    SystemMessage(content="You are MediGuide, a friendly health information assistant."),
    HumanMessage(content="What causes a sore throat?")
])
print(response.content)
```

---

##  Example Queries & Outputs

### Query 1: General Symptom (Required by Task)
```
  User: What causes a sore throat?

 MediGuide:
A sore throat is usually caused by one of these common reasons:

• Viral infections — The most common cause (cold, flu, COVID-19)
• Bacterial infections — Strep throat is the most well-known
• Dry air — Especially in winter or air-conditioned rooms
• Allergies — Pollen, dust, or pet dander can irritate the throat
• Acid reflux — Stomach acid can reach and irritate the throat
• Overuse of voice — Talking/shouting too much

Most sore throats from viral infections resolve within 5-7 days.

If your sore throat is severe, lasts more than a week, or is
accompanied by high fever, please consult a healthcare professional.
```

---

### Query 2: OTC Medication Safety (Required by Task)
```
 User: Is paracetamol safe for children?

 MediGuide:
Paracetamol (also called acetaminophen) is generally safe for children
when used correctly. Here are the key guidelines:

• Always use the correct dose based on the child's WEIGHT, not age
• Children's paracetamol comes in liquid (syrup) form — easier to dose
• Standard dose: 10–15 mg per kg of body weight
• Do not give more than 4 doses in 24 hours
• Do not give to infants under 2 months without a doctor's advice
• Never use adult paracetamol tablets for children — too high a dose

 Overdose risk: Always read the label and do not exceed the
recommended amount, as too much paracetamol can harm the liver.

 For your child's specific situation, please consult a pediatrician
or pharmacist who can recommend the exact dose.
```

---

### Query 3: Emergency Detection (Safety Filter Demo)
```
 User: I have severe chest pain and can't breathe

 MediGuide:
 EMERGENCY — Please Act Immediately!

The symptoms you described may need urgent medical attention right now.

 Call Emergency Services Immediately:
• 🇵🇰 Pakistan Rescue : 115
• 🇵🇰 Emergency      : 1122
•  International   : 911 or your local emergency number

⚠️ Please do NOT wait — call for help immediately!
```
> Note: This response was generated by the Python safety filter — the LLM was never called.

---

### Query 4: Diagnosis Request (Refused)
```
 User: I'm always thirsty. Do I have diabetes?

 MediGuide: [ HIGH-RISK — Extra caution applied]
I understand your concern, and it's good that you're paying attention
to your body's signals. However, I'm not able to diagnose any medical
condition — only a licensed doctor can do that after a proper examination
and blood tests.

Persistent thirst can be caused by many things:
• Dehydration — not drinking enough water
• High sugar or salt intake
• Certain medications
• Diabetes (Type 1 or Type 2) — among other possibilities
• Dry mouth conditions

What I recommend:
1. Track how much water you drink daily
2. Note any other symptoms (frequent urination, fatigue, blurry vision)
3. Share this information with your doctor

 Please visit a healthcare professional for proper evaluation
and blood glucose testing. Early detection is very important.
```

---

##  Multi-turn Conversation

MediGuide remembers the full conversation history using LangChain's message system. This enables natural, context-aware dialogue:

```
 Turn 1: "I've been having headaches for the past 3 days."
MediGuide: [Explains common causes of headaches...]

 Turn 2: "They happen mostly in the morning after waking up."
 MediGuide: [Connects to morning-specific causes like dehydration,
               sleep position, blood pressure — remembers "headaches"
               from Turn 1 without needing to repeat it]

 Turn 3: "Could this be related to dehydration or screen time?"
 MediGuide: [Directly addresses both factors in context of the
               morning headache pattern already established]

 Turn 4: "When should I worry and see a doctor about this?"
 MediGuide: [Gives specific red flags relevant to the full
               headache context discussed so far]
```

This is made possible by maintaining a `chat_history` list of `HumanMessage` and `AIMessage` objects that are sent with every new request.

---

##  Prompt Engineering Analysis

The notebook includes a side-by-side comparison of three prompt versions using the **same query** and the **same model** (`llama-3.3-70b-versatile`):

### Test Query: *"Is paracetamol safe for children?"*

| Aspect | V1 — Minimal | V2 — Role + Tone | V3 — Full Engineering |
|--------|-------------|-----------------|----------------------|
| Prompt Size | 22 chars | 180 chars | 1,100+ chars |
| Safety Disclaimer |  None | Sometimes |  Always |
| Dosage Guidance | Generic | General | Specific with warnings |
| Doctor Referral |  Missing | Sometimes |  Always present |
| Response Structure | Unstructured | Some bullets |  Clear bullets |
| Emergency Handling |  None |  None |  Defined rules |
| Tone | Robotic | Friendly | Warm & Professional |
| **Overall Quality** |  Poor | ~ Adequate |  Excellent |

**Conclusion:** Full prompt engineering (V3) produces dramatically safer, more structured, and more helpful responses from the same underlying model.

---

##  Project Structure

```
task4-health-chatbot/
│
├── 📓 Task4_Health_Query_Chatbot.ipynb    # Main Jupyter Notebook
│     ├── Section 1: Installation & Imports
│     ├── Section 2: Groq API Configuration
│     ├── Section 3: System Prompt Design
│     ├── Section 4: Safety Filter Module
│     ├── Section 5: HealthChatbot Class (ChatGroq)
│     ├── Section 6: Testing — 7 Example Queries
│     ├── Section 7: Multi-turn Conversation Demo
│     ├── Section 8: Prompt Engineering Comparison
│     └── Section 9: Results & Final Insights
│
├── 📄 README.md                           # This file
├── 🔒 .env                               # API keys (NOT committed to GitHub)
├── 📋 .env.example                       # Template for .env file
├── 📦 requirements.txt                   # Python dependencies
└── 🚫 .gitignore                         # Excludes .env and cache files
```

### requirements.txt
```
groq>=0.9.0
langchain-groq>=0.1.0
langchain-core>=0.2.0
python-dotenv>=1.0.0
ipykernel>=6.0.0
jupyter>=1.0.0
```

### .env.example
```
# Copy this file to .env and fill in your actual API key
# Get a free key at: https://console.groq.com

GROQ_API_KEY=gsk_your_groq_api_key_here
```

### .gitignore
```
.env
__pycache__/
*.pyc
.ipynb_checkpoints/
*.egg-info/
```

---

##  Key Results & Findings

### 1. Prompt Engineering Impact
- **V3 (Full Engineering)** produced 100% disclaimer inclusion vs 0% for V1
- Constraint injection (`NEVER diagnose`) was more effective than soft guidelines
- Role prompting created consistent persona across all conversation turns

### 2. Safety System Performance
- **Emergency detection accuracy:** 100% on tested patterns
- **Diagnosis refusal rate:** 100% for flagged high-risk queries
- **False positive rate:** 0% — safe queries were not blocked

### 3. Model Performance (LLaMA 3.3 70B via Groq)
- **Response latency:** < 2 seconds per query (Groq's ultra-fast inference)
- **Response quality:** Professional, medically responsible, well-structured
- **Context retention:** Maintained full context across 4+ conversation turns

### 4. Key Technical Learnings

| # | Learning |
|---|---------|
| 1 | **Prompt engineering transforms LLM behavior** — same model, better prompts = much better outputs |
| 2 | **Multi-layer safety is essential** for health applications — never rely on LLM alone |
| 3 | **Groq's speed** (~800 tok/sec) makes conversational AI feel instant and natural |
| 4 | **LangChain message history** cleanly handles multi-turn conversation state |
| 5 | **Environment variables** (`.env`) are the professional way to manage API credentials |

---

##  Future Improvements

| Improvement | Description | Priority |
|-------------|-------------|----------|
| **Streamlit Web UI** | Beautiful web interface instead of CLI | High |
| **RAG Integration** | Connect to PubMed / WHO databases for cited answers | High |
| **Urdu Language Support** | For Pakistani users — translate prompts and responses | Medium |
| **Voice Input** | Use OpenAI Whisper for spoken health queries | Medium |
| **Symptom Checker** | Structured JSON output with possible causes ranked | Medium |
| **Response Logging** | Save all queries/responses for analysis and improvement | Low |
| **Fine-tuning** | Fine-tune on medical Q&A datasets for better accuracy | Low |

---

##  Author

**[Bilal Ahmed]**
AI/ML Engineering Intern — DevelopersHub Corporation

- 📧 Email: bilalahmadc.m56@gmail.com
- 🐙 GitHub: [@BilalRajput-52](https://github.com/BilalRajput-52)
- 💼 LinkedIn: [linkedin.com/in/bilal-ahmed56ab](https://linkedin.com/in/bilal-ahmed56ab)

---

 


##  Disclaimer

> **MediGuide is for general educational purposes only.**
> It does not provide medical advice, diagnosis, or treatment.
> The information provided is not a substitute for professional medical advice.
> Always seek the guidance of a qualified healthcare provider with any questions
> you may have regarding a medical condition.
> **In case of a medical emergency, call 115 (Pakistan) or your local emergency number immediately.**

---

<div align="center">

Made with love for the DevelopersHub Corporation AI/ML Internship

 If this project helped you, please give it a star on GitHub!

</div>
