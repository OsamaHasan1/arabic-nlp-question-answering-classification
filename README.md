# Arabic NLP Question Answering & Classification System

This project is an Arabic Natural Language Processing system that works with Arabic questions and answers. The system includes Arabic text preprocessing, question category classification, Arabic question answering, machine translation, and a Streamlit application for deployment.

The main goal of this project is to build a complete Arabic NLP pipeline where the user enters an Arabic question, and the system generates an Arabic answer, predicts the question category, and translates the output into English.

---

## Deployment Demo

The GIF below shows the deployed Streamlit application. It demonstrates how the user enters an Arabic question and how the system returns an Arabic answer, predicted category, and English translation.

![Deployment Demo](assets/deployment_demo.gif)

---

## Project Overview

Arabic text processing is challenging because Arabic contains different writing forms, diacritics, tatweel, punctuation, stopwords, and rich morphology. These characteristics can affect how machine learning and deep learning models understand Arabic text.

The project was divided into five main stages:

1. **Arabic Text Preprocessing**
2. **Arabic Question Classification**
3. **Arabic Question Answering**
4. **Machine Translation**
5. **Streamlit Deployment**

Each stage was implemented in a separate notebook or file to make the workflow easier to understand, test, and evaluate.

---

## Repository Structure

```text
arabic-nlp-question-answering-classification/
│
├── README.md
├── requirements.txt
├── .gitignore
├── app.py
│
├── arabic_nlp_preprocessing.ipynb
├── arabic_question_classification.ipynb
├── arabic_question_answering.ipynb
├── arabic_machine_translation.ipynb
│
├── data/
│   ├── README.md
│   ├── AAFAQ_Dataset.csv
│   └── cleaned_data.csv
│
├── models/
│   └── README.md
│
└── assets/
    └── deployment_demo.gif
```

---

## Dataset

The project uses the **AAFAQ Arabic question-answering dataset**.

The dataset contains Arabic questions, answers, and category labels. It was used for multiple NLP tasks, including preprocessing, classification, question answering, translation experiments, and Streamlit application testing.

The dataset files are stored inside the `data/` folder:

```text
data/
├── AAFAQ_Dataset.csv
└── cleaned_data.csv
```

* `AAFAQ_Dataset.csv`: The original Arabic dataset.
* `cleaned_data.csv`: The cleaned version after applying Arabic preprocessing.

---

# 1. Arabic Text Preprocessing

## File

```text
arabic_nlp_preprocessing.ipynb
```

## What Was Done

The preprocessing notebook was used to clean and prepare the Arabic dataset before using it in the classification and question-answering stages.

Arabic text can contain noise such as diacritics, tatweel, punctuation marks, repeated spaces, and common stopwords. These elements can make the same word appear in different forms, which makes it harder for machine learning models to learn useful patterns.

The final output of this stage was saved as:

```text
data/cleaned_data.csv
```

---

## Why Preprocessing Was Needed

Preprocessing was needed to make the Arabic text cleaner and more consistent.

For traditional machine learning models such as Bag of Words, TF-IDF, Logistic Regression, and SVM, cleaner text helps reduce unnecessary vocabulary variation. This makes it easier for the model to focus on important words related to the question category.

However, Arabic question answering needs more natural text because removing too many words can damage the meaning of the question. For that reason, the project later used stronger preprocessing for classification and lighter preprocessing for question answering.

---

## Preprocessing Steps

### 1. Removing Tashkeel

Arabic diacritics such as fatha, damma, kasra, sukoon, and tanween were removed.

Example:

```text
السَّلامُ عليكم
```

becomes:

```text
السلام عليكم
```

This helped reduce unnecessary variations in the same word.

---

### 2. Removing Tatweel

Tatweel characters were removed because they are decorative and do not add meaning.

Example:

```text
الســــلام
```

becomes:

```text
السلام
```

This made the text more consistent without changing the meaning.

---

### 3. Removing Punctuation

Punctuation marks such as commas, dots, question marks, and other symbols were removed.

Example:

```text
ما هي فوائد التعليم؟
```

becomes:

```text
ما هي فوائد التعليم
```

This reduced noise in the text and helped traditional vectorization methods.

---

### 4. Removing Stopwords

Common Arabic stopwords were removed, such as:

```text
من، في، على، عن، إلى، هل، ما، هذا، هذه
```

Stopword removal helped classification models focus on important topic-related words.

However, this step can be risky for question answering because words such as “ما”, “هل”, and “كيف” can affect the meaning of a question. Therefore, the final app keeps the question more natural for the QA model.

---

### 5. Tokenization

Tokenization was used to split Arabic text into words during preprocessing. This was useful for operations such as stopword removal and text cleaning.

The final cleaned data was saved as readable text rather than token lists, because the later models needed complete sentence input.

---

### 6. Removing Extra Spaces

Extra spaces were removed after cleaning.

Example:

```text
ما    فوائد     التعليم
```

becomes:

```text
ما فوائد التعليم
```

---

## Techniques Not Used in the Final Cleaning

Some preprocessing techniques were tested or considered but were not used in the final cleaned dataset.

### Hamza Normalization

Hamza normalization was not used because it made the output harder to read in some cases.

### Stemming

Stemming was avoided because Arabic stemming can be aggressive and may change words into unnatural root-like forms. This can damage readability, especially for question answering.

### Lemmatization

Lemmatization was not used because it caused technical issues and was not stable enough for the final preprocessing pipeline.

---

## Preprocessing Evaluation

The final preprocessing approach successfully reduced surface noise in the Arabic text while preserving the main meaning. Removing diacritics and tatweel was especially useful because it standardized the text without damaging readability.

The main limitation was that one preprocessing pipeline cannot perfectly serve both classification and question answering. Classification benefits from stronger cleaning, while question answering needs more natural and complete Arabic input.

For this reason, the final application uses:

* Cleaned text for classification.
* Natural Arabic text for question answering.

---

# 2. Arabic Question Classification

## File

```text
arabic_question_classification.ipynb
```

## What Was Done

The classification notebook was used to train and compare models that classify Arabic questions into predefined categories.

The target was to predict the category of a given Arabic question.

Example:

```text
ما هي اهمية التعليم
```

Expected category:

```text
التعليم
```

---

## Classification Categories

The project used 17 Arabic categories:

```text
الاقتصاد والعمل
البيئة والطاقة
البيولوجيا
التاريخ
الترفيه
التطوع
التعليم
التكنولوجيا
الثقافة
الجغرافيا
الدين
الرياضة
السفر والسياحة
السياسة والقانون
الصحة
العلوم
علم الاجتماع
```

The category labels were converted into numerical labels from 0 to 16 so that the models could learn from them.

---

## Models Tested

Several traditional and neural approaches were tested:

* Bag of Words + Logistic Regression
* Bag of Words + SVM
* TF-IDF + Logistic Regression
* TF-IDF + SVM
* BERT Embeddings + Logistic Regression
* BERT Embeddings + SVM
* Fine-tuned AraBERT

---

## Bag of Words

Bag of Words converts text into numerical features by counting word occurrences. It is simple and useful as a baseline, but it ignores word order, grammar, and context.

### Bag of Words Results

| Model                              | Accuracy | F1-score |
| ---------------------------------- | -------: | -------: |
| Bag of Words + Logistic Regression |   0.6088 |   0.6098 |
| Bag of Words + SVM                 |   0.5898 |   0.5901 |

### Interpretation

Bag of Words produced the weakest results. This is because it depends only on word frequency and does not understand context or meaning. Since Arabic questions can have similar meanings with different wording, Bag of Words was not strong enough for this task.

---

## TF-IDF

TF-IDF gives higher importance to words that are important in a specific question but not too common across the dataset.

### TF-IDF Results

| Model                        | Accuracy | F1-score |
| ---------------------------- | -------: | -------: |
| TF-IDF + Logistic Regression |   0.6307 |   0.6225 |
| TF-IDF + SVM                 |   0.6587 |   0.6554 |

### Interpretation

TF-IDF performed better than Bag of Words because it focused more on important category-related words.

TF-IDF + SVM was the best traditional machine learning model. SVM worked well with TF-IDF because TF-IDF produces high-dimensional sparse features, and SVM is suitable for separating classes in this type of feature space.

However, TF-IDF still cannot fully understand Arabic meaning, context, or word relationships.

---

## BERT Embeddings

BERT embeddings represent the meaning of text using dense contextual vectors. Instead of simply counting words, BERT embeddings can capture more semantic information from the question.

### BERT Embedding Results

| Model                                 | Accuracy | F1-score |
| ------------------------------------- | -------: | -------: |
| BERT Embeddings + Logistic Regression |   0.6986 |   0.6963 |
| BERT Embeddings + SVM                 |   0.7146 |   0.7122 |

### Interpretation

BERT embeddings performed better than Bag of Words and TF-IDF because they captured more meaning and context.

BERT + SVM performed better than BERT + Logistic Regression, which suggests that SVM handled the embedding space better in this experiment.

However, BERT embeddings were still used as fixed feature vectors, so the BERT model itself was not fully adapted to the AAFAQ classification task.

---

## Fine-tuned AraBERT

AraBERT is a transformer-based model designed specifically for Arabic text. It was fine-tuned directly on the classification task.

### AraBERT Results

| Model              | Accuracy | F1-score |
| ------------------ | -------: | -------: |
| Fine-tuned AraBERT |   0.7445 |   0.7401 |

### Interpretation

AraBERT achieved the best classification performance.

It performed better because it was trained directly on the AAFAQ categories and could adapt its weights to the vocabulary and patterns of the dataset. Unlike Bag of Words and TF-IDF, AraBERT can understand Arabic context and relationships between words.

---

## Final Classification Comparison

| Model                                 | Accuracy | F1-score |
| ------------------------------------- | -------: | -------: |
| Bag of Words + Logistic Regression    |   0.6088 |   0.6098 |
| Bag of Words + SVM                    |   0.5898 |   0.5901 |
| TF-IDF + Logistic Regression          |   0.6307 |   0.6225 |
| TF-IDF + SVM                          |   0.6587 |   0.6554 |
| BERT Embeddings + Logistic Regression |   0.6986 |   0.6963 |
| BERT Embeddings + SVM                 |   0.7146 |   0.7122 |
| Fine-tuned AraBERT                    |   0.7445 |   0.7401 |

---

## Best Classification Model

The best classification model was:

```text
Fine-tuned AraBERT
```

AraBERT was selected for the final Streamlit application because it achieved the highest accuracy and F1-score among all classification models.

---

# 3. Arabic Question Answering

## File

```text
arabic_question_answering.ipynb
```

## What Was Done

The question-answering notebook was used to train and compare models that generate Arabic answers for Arabic questions.

The goal was to take an Arabic question as input and generate a relevant Arabic answer.

---

## QA Models Tested

Several models were tested:

* TF-IDF + RNN Seq2Seq
* BERT Embeddings + RNN Seq2Seq
* mT5
* AraGPT2
* Qwen2.5-0.5B

The models were evaluated using BLEU score.

---

## TF-IDF + RNN Seq2Seq

This model used TF-IDF vectors as input to an RNN sequence-to-sequence model.

### Result

| Model                | BLEU Score |
| -------------------- | ---------: |
| TF-IDF + RNN Seq2Seq |    0.00197 |

### Interpretation

This model performed very poorly. The main reason is that TF-IDF does not preserve word order or sentence structure. Since RNN models depend on sequential information, using TF-IDF weakened the input representation.

This model was useful as a baseline experiment, but it was not suitable for generating meaningful Arabic answers.

---

## BERT Embeddings + RNN Seq2Seq

This model used BERT embeddings as input features for an RNN Seq2Seq model.

### Result

| Model                         | BLEU Score |
| ----------------------------- | ---------: |
| BERT Embeddings + RNN Seq2Seq |    0.00624 |

### Interpretation

This model improved slightly compared with TF-IDF + RNN because BERT embeddings contain more contextual meaning.

However, the BLEU score was still very low. This means the generated answers were still far from the reference answers. The model struggled because BERT was used only as a fixed feature extractor, and the RNN model still had limitations in generating strong Arabic answers.

---

## mT5

mT5 is a multilingual text-to-text transformer model. It is more suitable for question answering than RNN baselines because it uses self-attention and can better understand the full question context.

### Result

| Model | BLEU Score |
| ----- | ---------: |
| mT5   |    0.06817 |

### Interpretation

mT5 showed clear improvement over the RNN-based models. This indicates that transformer-based architectures are more suitable for Arabic answer generation than traditional RNN approaches.

However, the BLEU score was still relatively low, so the model was not the strongest QA model in the project.

---

## AraGPT2

AraGPT2 is an Arabic generative transformer model. It was used because it is designed for Arabic text generation.

### Result

| Model   | BLEU Score |
| ------- | ---------: |
| AraGPT2 |    0.09807 |

### Interpretation

AraGPT2 performed better than mT5 in this experiment. Since AraGPT2 is designed for Arabic generation, it was more suitable than the previous models for producing Arabic answers.

However, the result still showed that Arabic question answering is difficult, especially when the model must generate answers that closely match reference answers.

---

## Qwen2.5-0.5B

Qwen2.5-0.5B is a transformer-based language model used for Arabic answer generation in this project.

### Result

| Model        | BLEU Score |
| ------------ | ---------: |
| Qwen2.5-0.5B |    0.20851 |

### Interpretation

Qwen2.5-0.5B achieved the best BLEU score among all question-answering models.

It performed better because it is a stronger generative model and can produce more fluent and meaningful answers compared with the RNN-based baselines and previous transformer models.

---

## Final QA Comparison

| Model                         | BLEU Score |
| ----------------------------- | ---------: |
| TF-IDF + RNN Seq2Seq          |    0.00197 |
| BERT Embeddings + RNN Seq2Seq |    0.00624 |
| mT5                           |    0.06817 |
| AraGPT2                       |    0.09807 |
| Qwen2.5-0.5B                  |    0.20851 |

---

## Best QA Model

The best question-answering model was:

```text
Qwen2.5-0.5B
```

Qwen was selected for the final Streamlit application because it achieved the highest BLEU score and produced the best Arabic answer generation results among the tested QA models.

---

# 4. Machine Translation

## File

```text
arabic_machine_translation.ipynb
```

## What Was Done

The machine translation part was used to translate Arabic output into English.

The purpose of this stage was to make the system more understandable for users who do not read Arabic. After generating an Arabic answer and predicting the Arabic category, the output can be translated into English.

---

## Why Machine Translation Was Added

The project mainly works with Arabic questions and Arabic answers. However, adding translation makes the system more accessible and easier to demonstrate to a wider audience.

Machine translation helps show:

* The original Arabic question.
* The generated Arabic answer.
* The predicted Arabic category.
* The English translation of the result.

---

## Translation in the Final App

In the Streamlit application, translation is handled through an external API. The app sends the Arabic question, generated answer, and predicted category to the translation model, then displays the English version.

The translated output follows this structure:

```text
Question:
Answer:
Category:
```

This makes the output clear and easy to read.

---

## Important Security Note

The translation API key should not be written directly inside the code when uploading the project to GitHub.

A safer approach is to use:

* Environment variables
* Streamlit secrets

Example environment variable name:

```text
OPENROUTER_API_KEY
```

The `.gitignore` file should exclude secret files such as:

```text
.env
.streamlit/secrets.toml
```

---

# 5. Streamlit Deployment

## File

```text
app.py
```

## What the Streamlit App Does

The Streamlit file connects the main parts of the project into one interactive web application.

The app allows the user to enter an Arabic question, then it performs three main tasks:

1. Generates an Arabic answer.
2. Predicts the Arabic question category.
3. Translates the result into English.

The app displays:

* Original Question
* Generated Arabic Answer
* Predicted Category
* Translated Output

---

## Why Streamlit Was Used

Streamlit was used because it makes it easy to deploy machine learning and NLP projects as interactive web applications using Python.

It allows the project to be demonstrated clearly without requiring users to run each notebook separately.

Instead of showing only code and results, Streamlit provides a simple interface where users can test the Arabic NLP system directly.

---

## Models Used in the App

The final Streamlit app uses two main local model folders:

```text
arabert_classification_model/
qwen_qa_model/
```

---

## Why AraBERT Was Used for Classification

AraBERT was used in the app because it achieved the best classification performance in the project.

It is designed for Arabic language understanding and was fine-tuned on the project categories. This made it more suitable than traditional models such as Bag of Words, TF-IDF, Logistic Regression, and SVM.

The app uses AraBERT to predict the category of the Arabic question.

---

## Why Qwen Was Used for Question Answering

Qwen2.5-0.5B was used in the app because it achieved the best question-answering result in the project.

It generated better Arabic answers than the RNN baselines, mT5, and AraGPT2 according to BLEU score.

The app uses Qwen to generate the Arabic answer from the user question.

---

## Different Preprocessing for Different Tasks

The app uses two preprocessing strategies:

### Classification Preprocessing

For classification, the question is cleaned using Arabic preprocessing steps such as removing diacritics, tatweel, punctuation, stopwords, and extra spaces.

This helps the classification model focus on important category-related words.

### QA Preprocessing

For question answering, the app keeps the Arabic question more natural and complete.

This is important because the QA model needs to understand the full question meaning before generating an answer.

---

## Expected Local Model Structure

To run the app locally, the trained model folders should be placed beside `app.py`:

```text
arabic-nlp-question-answering-classification/
│
├── app.py
├── arabert_classification_model/
├── qwen_qa_model/
└── requirements.txt
```

The model folders are not included in the repository because they are large. More details are provided in:

```text
models/README.md
```

---

# How to Run the Project

## 1. Clone the Repository

```bash
git clone https://github.com/OsamaHasan1/arabic-nlp-question-answering-classification.git
```

## 2. Move to the Project Folder

```bash
cd arabic-nlp-question-answering-classification
```

## 3. Install Requirements

```bash
pip install -r requirements.txt
```

## 4. Add the Required Model Folders

Place the trained model folders beside `app.py`:

```text
arabert_classification_model/
qwen_qa_model/
```

## 5. Add Translation API Key

For local testing, set the API key as an environment variable.

On Windows CMD:

```bash
set OPENROUTER_API_KEY=your_key_here
```

Or use Streamlit secrets locally:

```text
.streamlit/secrets.toml
```

Inside the file:

```toml
OPENROUTER_API_KEY = "your_key_here"
```

Do not upload `.streamlit/secrets.toml` to GitHub.

## 6. Run the Streamlit App

```bash
streamlit run app.py
```

If Streamlit is not recognized, use:

```bash
python -m streamlit run app.py
```

---

# Results Summary

## Classification Results

| Model                                 | Accuracy | F1-score |
| ------------------------------------- | -------: | -------: |
| Bag of Words + Logistic Regression    |   0.6088 |   0.6098 |
| Bag of Words + SVM                    |   0.5898 |   0.5901 |
| TF-IDF + Logistic Regression          |   0.6307 |   0.6225 |
| TF-IDF + SVM                          |   0.6587 |   0.6554 |
| BERT Embeddings + Logistic Regression |   0.6986 |   0.6963 |
| BERT Embeddings + SVM                 |   0.7146 |   0.7122 |
| Fine-tuned AraBERT                    |   0.7445 |   0.7401 |

Best classification model:

```text
Fine-tuned AraBERT
```

---

## Question Answering Results

| Model                         | BLEU Score |
| ----------------------------- | ---------: |
| TF-IDF + RNN Seq2Seq          |    0.00197 |
| BERT Embeddings + RNN Seq2Seq |    0.00624 |
| mT5                           |    0.06817 |
| AraGPT2                       |    0.09807 |
| Qwen2.5-0.5B                  |    0.20851 |

Best QA model:

```text
Qwen2.5-0.5B
```

---

# Limitations

This project has several limitations:

* The QA models sometimes generate weak or inaccurate answers.
* BLEU scores for the QA task were generally low, which shows that Arabic answer generation is difficult.
* The final QA model still requires improvement before being used in real-world applications.
* The project depends on local model folders that are not uploaded to GitHub because of size.
* The translation part depends on an external API.
* The Streamlit app requires the classification and QA model folders to exist locally.
* A single preprocessing method is not ideal for both classification and question answering.
* The classification model performed better than traditional approaches, but there is still room for improvement.

---

# Future Improvements

Future improvements could include:

* Training the QA model for more epochs.
* Using a larger and more diverse Arabic QA dataset.
* Improving Arabic preprocessing separately for each NLP task.
* Trying stronger Arabic language models.
* Adding better evaluation metrics for generated answers.
* Deploying the model using cloud hosting.
* Storing large model files using external model hosting platforms.
* Improving the Streamlit app interface.
* Adding confidence scores for classification.
* Adding more examples and test cases.

---

# Tools and Libraries

The project was built using:

* Python
* Streamlit
* PyTorch
* Transformers
* Requests
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* NLTK
* PyArabic
* Emoji
* SentencePiece
* Accelerate
* Jupyter Notebook

---

# Conclusion

This project built a complete Arabic NLP pipeline for question answering and classification.

The preprocessing stage cleaned Arabic text and produced a more consistent dataset. The classification stage compared traditional and neural models, where fine-tuned AraBERT achieved the best classification performance. The question-answering stage compared several generation models, where Qwen2.5-0.5B achieved the best BLEU score. The machine translation stage allowed Arabic outputs to be translated into English. Finally, the Streamlit app connected the best classification and question-answering models into one interactive interface.

Overall, the project demonstrates how Arabic NLP models can be combined into a practical application that generates answers, predicts question categories, and translates results for easier understanding.
