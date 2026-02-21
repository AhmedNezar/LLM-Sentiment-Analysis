# 🧠 LLM Sentiment Analysis (Gemini 2.5 Flash)

A structured Large Language Model (LLM) powered sentiment analysis system built using Google Gemini 2.5 Flash and LangChain.

This project demonstrates prompt engineering techniques, structured reasoning, strict JSON enforcement, and few-shot learning for reliable sentiment classification.

---

## 🚀 Overview

This system analyzes input text and classifies it into one of four sentiment labels:

- Positive
- Negative
- Neutral
- Mixed

It supports two operating modes:

1. Reasoning Mode → Returns step-by-step structured reasoning + final sentiment
2. Label-Only Mode → Returns only the final sentiment (faster & lightweight)

The output is always enforced as valid JSON and saved to a file.

---

## 🧱 Tech Stack

- Python
- LangChain
- Google Generative AI (Gemini 2.5 Flash)
- python-dotenv
- JSON

---

## 📦 Installation

Install required packages:

```bash
pip install langchain langchain-google-genai python-dotenv
```

---

## 🔑 Environment Configuration

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

The project loads environment variables using:

```python
from dotenv import load_dotenv
load_dotenv()
```

---

## 📂 Project Structure

```
.
├── Task1.py
├── output.json
└── .env
```

---

## ⚙️ How It Works

### 1️⃣ Model Initialization

```python
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash")
```

---

### 2️⃣ analyze() Function

```python
def analyze(text, reasoning=False):
```

Parameters:

- text (str) → Input text to analyze
- reasoning (bool) → Enable structured reasoning mode

The function dynamically selects between two prompts:

- reasoning_prompt
- label_prompt

Then:

1. Sends prompt to Gemini
2. Parses strict JSON response
3. Saves result to `Tasks/output.json`

---

## 🧠 Reasoning Mode

When `reasoning=True`, the model performs structured internal reasoning:

1. Identify the main subject
2. Detect emotionally charged words
3. Evaluate polarity
4. Consider negations and intensity
5. Detect contrast words (e.g., but, however)
6. Weigh overall tone
7. Decide final sentiment

### Output Format

```json
{
  "reasoning": "Step-by-step explanation of how the sentiment was determined.",
  "final_sentiment": "Positive | Negative | Neutral | Mixed"
}
```

---

## ⚡ Label-Only Mode

When `reasoning=False`, the model returns only:

```json
{
  "final_sentiment": "Positive | Negative | Neutral | Mixed"
}
```

This mode is faster and cheaper for production usage.

---

## ▶️ Example Usage

```python
analyze("It was very good really", True)
```

---

## 📤 Example

### Input

"I love the design, but the battery life is terrible."

### Output

```json
{
  "reasoning": "The text contains both positive and negative expressions...",
  "final_sentiment": "Mixed"
}
```

---

## 🛡 Error Handling

The function wraps execution in a try/except block to catch:

- Invalid JSON responses
- API errors
- Model formatting issues

If parsing fails, an error message is printed.

---

## 📌 Key Design Decisions

- Few-shot prompting improves classification consistency
- Strict JSON enforcement prevents malformed outputs
- Separation between reasoning and classification modes
- Output persistence for evaluation & debugging

---
