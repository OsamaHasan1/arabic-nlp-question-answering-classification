# Dataset

This folder contains the dataset files used in the Arabic NLP project.

## Files

* `AAFAQ_Dataset.csv`: The original Arabic dataset used in the project.
* `cleaned_data.csv`: The cleaned version of the dataset after applying Arabic text preprocessing.

---

## Dataset Description

The dataset contains Arabic question-answering data. It includes Arabic questions, their answers, and category labels.

The dataset was used for several NLP tasks:

* Arabic text preprocessing
* Question category classification
* Arabic question answering
* Machine translation experiments
* Streamlit application testing

---

## Why Data Cleaning Was Needed

Arabic text can contain many elements that make NLP processing harder, such as diacritics, tatweel, punctuation, extra spaces, and common stopwords.

Before using the data for classification and question answering, the text needed to be cleaned to make it more consistent and easier for machine learning and deep learning models to process.

---

## Cleaning Steps

To create `cleaned_data.csv`, several preprocessing steps were applied to the original dataset.

### 1. Removing Arabic Diacritics

Arabic diacritics such as fatha, damma, kasra, and sukoon were removed.

Example:

```text
السَّلامُ عليكم
```

becomes:

```text
السلام عليكم
```

This helps reduce unnecessary text variation.

---

### 2. Removing Tatweel

Tatweel characters were removed from Arabic words.

Example:

```text
الســــلام
```

becomes:

```text
السلام
```

This makes the words cleaner and more consistent.

---

### 3. Removing Punctuation

Punctuation marks such as commas, question marks, dots, and symbols were removed.

Example:

```text
ما هي فوائد التعليم؟
```

becomes:

```text
ما هي فوائد التعليم
```

This helps the model focus more on the actual words instead of symbols.

---

### 4. Removing Stopwords

Some common Arabic stopwords were removed, such as:

```text
من، في، على، عن، إلى، هل، ما، هذا، هذه
```

Stopwords are very frequent words that may not add strong meaning for classification tasks.

---

### 5. Removing Extra Spaces

After cleaning, extra spaces were removed and the text was normalized into a cleaner format.

Example:

```text
ما    فوائد     التعليم
```

becomes:

```text
ما فوائد التعليم
```

---

## Notes About Preprocessing Decisions

Not every Arabic preprocessing technique was used.

Some techniques were avoided because they damaged the meaning or readability of the Arabic text.

For example:

* Aggressive stemming was avoided because it can change Arabic words into unnatural root forms.
* Lemmatization was not used because it caused technical issues and was not stable in the project.
* Heavy normalization was avoided when it made the text harder to understand.

The final cleaning process focused on safe preprocessing steps that improved text quality without destroying the meaning of the Arabic questions and answers.

---

## Final Output

The final cleaned dataset was saved as:

```text
cleaned_data.csv
```

This cleaned file was then used in later stages of the project, including:

* Arabic question classification
* Question-answering model training
* Evaluation experiments
* Streamlit app testing

---

## Important Note

The original dataset and cleaned dataset are included for educational and experimental purposes.

If the dataset files are missing, place them inside this folder using the following structure:

```text
data/
├── AAFAQ_Dataset.csv
└── cleaned_data.csv
```

Then update the notebook paths if needed.
