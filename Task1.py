from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import json

load_dotenv()

llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash")


def analyze(text, reasoning=False):
    reasoning_prompt = """
    You are an expert sentiment analysis engine.

    Your task is to analyze the sentiment of the given text using structured step-by-step reasoning.

    Follow this reasoning process internally:
    1. Identify the main subject.
    2. Detect emotionally charged words or phrases.
    3. Evaluate polarity (positive, negative, neutral).
    4. Consider negations and intensity modifiers.
    5. Detect contrast words (e.g., but, however), sarcasm, or mixed emotions.
    6. Weigh overall tone.
    7. Decide the final sentiment label.

    Sentiment labels (choose ONE only):
    - Positive
    - Negative
    - Neutral
    - Mixed

    IMPORTANT:
    - Return ONLY valid JSON.
    - Do NOT include markdown.
    - Do NOT include explanations outside JSON.
    - Output must strictly follow the specified format.

    Output format:

    {{
    "reasoning": "Step-by-step explanation of how the sentiment was determined.",
    "final_sentiment": "Positive | Negative | Neutral | Mixed"
    }}

    ----------------------------------
    Few-shot Examples:

    Example 1:
    Text: "I love the design, but the battery life is terrible."

    Output:
    {{
    "reasoning": "The text contains both positive and negative expressions. 'love the design' indicates positive sentiment. 'battery life is terrible' indicates strong negative sentiment. The contrast word 'but' shifts emphasis toward the negative aspect. Since both sentiments are clearly present, the overall sentiment is Mixed.",
    "final_sentiment": "Mixed"
    }}

    Example 2:
    Text: "The service was absolutely fantastic and the staff were very friendly."

    Output:
    {{
    "reasoning": "'absolutely fantastic' and 'very friendly' are strong positive expressions. There are no negative or contrasting elements. The overall tone is clearly positive.",
    "final_sentiment": "Positive"
    }}

    Example 3:
    Text: "The package arrived."

    Output:
    {{
    "reasoning": "The sentence is purely factual and contains no emotionally charged language. There are no positive or negative indicators. Therefore, the sentiment is Neutral.",
    "final_sentiment": "Neutral"
    }}

    ----------------------------------

    Now analyze the following text:

    Text:
    {}
    """.format(text)
    
    label_prompt = """
    You are an expert sentiment analysis engine.

    Your task is to classify the sentiment of the given text.

    Sentiment labels (choose ONE only):
    - Positive
    - Negative
    - Neutral
    - Mixed

    Classification Rules:
    - Positive → Overall favorable, happy, satisfied tone.
    - Negative → Overall critical, unhappy, or dissatisfied tone.
    - Neutral → Purely factual, no emotional tone.
    - Mixed → Clear presence of both positive and negative sentiment.

    IMPORTANT:
    - Return ONLY valid JSON.
    - Do NOT include explanations.
    - Do NOT include markdown.
    - Do NOT include any text outside JSON.
    - The output must strictly match the format below.

    Output format:

    {{
    "final_sentiment": "Positive | Negative | Neutral | Mixed"
    }}

    ----------------------------------
    Few-shot Examples:

    Example 1:
    Text: "I love the design, but the battery life is terrible."
    Output:
    {{
    "final_sentiment": "Mixed"
    }}

    Example 2:
    Text: "The service was absolutely fantastic."
    Output:
    {{
    "final_sentiment": "Positive"
    }}

    Example 3:
    Text: "The package arrived yesterday."
    Output:
    {{
    "final_sentiment": "Neutral"
    }}

    ----------------------------------

    Now classify:

    Text:
    {}
    """.format(text)

    prompt = reasoning_prompt if reasoning else label_prompt
    # print(prompt)
    try:
        response = llm.invoke(prompt)
        print(f"Response: {response.content}")
        result = json.loads(response.content)
        
        with open("output.json", 'w') as file:
            json.dump(result, file)
            
    except Exception as e:
        print(f"Error: {e}")
    
analyze("It was very good really", True)