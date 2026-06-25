import os
import re
from pathlib import Path

import torch
import streamlit as st
import requests

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM
)


# =========================
# Paths
# =========================

BASE_DIR = Path(__file__).resolve().parent

ARABERT_MODEL_PATH = BASE_DIR / "arabert_classification_model"
QWEN_MODEL_PATH = BASE_DIR / "qwen_qa_model"


# =========================
# Label names
# =========================

label_classes = [
    "الاقتصاد والعمل",
    "البيئة والطاقة",
    "البيولوجيا",
    "التاريخ",
    "الترفيه",
    "التطوع",
    "التعليم",
    "التكنولوجيا",
    "الثقافة",
    "الجغرافيا",
    "الدين",
    "الرياضة",
    "السفر والسياحة",
    "السياسة والقانون",
    "الصحة",
    "العلوم",
    "علم الاجتماع"
]


# =========================
# Page settings
# =========================

st.set_page_config(
    page_title="Arabic NLP System",
    page_icon="📘",
    layout="centered"
)

st.title("Arabic NLP Question Answering & Classification System")

st.write(
    "Enter an Arabic question. The system will generate an Arabic answer, "
    "predict the question category, and translate the output into English."
)


# =========================
# Arabic preprocessing
# =========================

arabic_stopwords = {
    "من", "في", "على", "عن", "إلى", "الى", "أن", "ان", "إن", "انه", "أنها",
    "هو", "هي", "هم", "هن", "هذا", "هذه", "ذلك", "تلك", "كان", "كانت",
    "يكون", "تكون", "و", "أو", "ثم", "كما", "قد", "لا", "ما", "هل"
}


def remove_tashkeel(text):
    text = str(text)
    arabic_diacritics = re.compile(r"[\u0617-\u061A\u064B-\u0652]")
    return re.sub(arabic_diacritics, "", text)


def remove_tatweel(text):
    text = str(text)
    return text.replace("ـ", "")


def remove_punctuation(text):
    text = str(text)

    punctuation = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~،؛؟"""

    for p in punctuation:
        text = text.replace(p, " ")

    return text


def remove_stopwords(text):
    text = str(text)
    words = text.split()
    clean_words = []

    for word in words:
        if word not in arabic_stopwords:
            clean_words.append(word)

    return " ".join(clean_words)


def preprocess_arabic_text(text):
    text = str(text)

    text = remove_tashkeel(text)
    text = remove_tatweel(text)
    text = remove_punctuation(text)
    text = remove_stopwords(text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_for_classification(question):
    return preprocess_arabic_text(question)


def preprocess_for_qa(question):
    # Qwen should receive the natural full Arabic question
    return str(question).strip()


# =========================
# Load models
# =========================

@st.cache_resource
def load_classification_model():
    tokenizer = AutoTokenizer.from_pretrained(ARABERT_MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(ARABERT_MODEL_PATH)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    model.eval()

    return tokenizer, model, device


@st.cache_resource
def load_qa_model():
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_PATH)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            QWEN_MODEL_PATH,
            torch_dtype=torch.float16
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            QWEN_MODEL_PATH,
            torch_dtype=torch.float32
        )

    model.to(device)
    model.eval()

    return tokenizer, model, device


classification_tokenizer, classification_model, classification_device = load_classification_model()
qa_tokenizer, qa_model, qa_device = load_qa_model()


# =========================
# Classification
# =========================

def classify_question(question):
    processed_question = preprocess_for_classification(question)

    inputs = classification_tokenizer(
        processed_question,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs = {
        key: value.to(classification_device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = classification_model(**inputs)
        logits = outputs.logits
        predicted_id = torch.argmax(logits, dim=1).item()

    predicted_category = label_classes[predicted_id]

    return predicted_category, processed_question


# =========================
# QA generation
# =========================

def generate_answer(question):
    processed_question = preprocess_for_qa(question)

    messages = [
        {
            "role": "system",
            "content": "أنت مساعد ذكي تجيب باللغة العربية بإجابة واضحة ومباشرة."
        },
        {
            "role": "user",
            "content": processed_question
        }
    ]

    try:
        prompt = qa_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    except Exception:
        prompt = f"السؤال: {processed_question}\nالإجابة:"

    inputs = qa_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    inputs = {
        key: value.to(qa_device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        output_ids = qa_model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
            pad_token_id=qa_tokenizer.eos_token_id
        )

    generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]

    answer = qa_tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    )

    answer = answer.strip()

    return answer, processed_question


# =========================
# Translation API
# =========================

def get_openrouter_key():
    return st.secrets.get("OPENROUTER_API_KEY", os.getenv("OPENROUTER_API_KEY"))


def translate_to_english(original_question, generated_answer, predicted_category):
    api_key = get_openrouter_key()

    if not api_key:
        return "Translation API key is missing. Add OPENROUTER_API_KEY to Streamlit secrets."

    text_to_translate = f"""
Arabic Question:
{original_question}

Arabic Generated Answer:
{generated_answer}

Arabic Predicted Category:
{predicted_category}
"""

    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "openai/gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": "Translate the Arabic text into clear English. Keep the structure: Question, Answer, Category."
            },
            {
                "role": "user",
                "content": text_to_translate
            }
        ],
        "temperature": 0
    }

    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=60
        )

        response.raise_for_status()

        result = response.json()

        return result["choices"][0]["message"]["content"].strip()

    except Exception as e:
        return f"Translation failed: {e}"


# =========================
# Interface
# =========================

question = st.text_area(
    "Enter Arabic Question",
    placeholder="اكتب السؤال العربي هنا...",
    height=120
)

if st.button("Run NLP System"):
    if question.strip() == "":
        st.warning("Please enter an Arabic question.")

    else:
        with st.spinner("Generating answer..."):
            generated_answer, qa_processed_question = generate_answer(question)

        with st.spinner("Classifying question..."):
            predicted_category, cls_processed_question = classify_question(question)

        with st.spinner("Translating output..."):
            translated_output = translate_to_english(
                question,
                generated_answer,
                predicted_category
            )

        st.subheader("Original Question")
        st.write(question)


        st.subheader("Generated Arabic Answer")
        st.write(generated_answer)


        st.subheader("Predicted Category")
        st.success(predicted_category)

        st.subheader("Translated Output")
        st.write(translated_output)