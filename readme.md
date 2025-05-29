# 🧠 Mental Health Chatbot Using LLMs

**Personal Project | LLM Fine-Tuning | NLP | Mental Health AI**

## 📌 Objective

To develop a **personalized Mental Health Chatbot** capable of responding with either:
- Logical reasoning (`T` mode)  
- Emotional support (`F` mode)

The chatbot was fine-tuned on **real-world counseling datasets** and follows a robust pipeline:
- Data cleaning  
- Zero-shot classification  
- Class balancing  
- OpenAI fine-tuning preparation  

---

## 📚 Datasets Used

- **Kaggle**  
  Mental Health Counseling Chatbot Dataset  
  → Provides general emotional support and conversational advice

- **Hugging Face**  
  CounselChat Dataset  
  → Contains expert responses written by licensed professionals

---

## 🧹 Data Preprocessing

✅ Combined **question** and **answer** pairs into single conversational samples  
✅ Cleaned non-ASCII characters, removed noisy tokens (e.g., emojis, excessive punctuation)  
✅ Used **NLTK** to filter out too-short or too-long responses  
✅ Applied **normal distribution filtering** (75th percentile cutoff)  
✅ Reduced dataset size:  
`4148` → `2796` meaningful samples  

---

## 🔍 Zero-Shot Labeling with BART

Used the `facebook/bart-large-mnli` model through:
```python
from transformers import pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
````

Labeled each answer as:

* **T** → Logical / analytical support
* **F** → Emotional / empathetic support

This enabled a **two-mode conversation system** where users can be matched to their preferred interaction style.

---

## ⚠️ Challenge: Class Imbalance

* Most original responses leaned toward **emotional (F)** support.
* To address this:

  * Removed ambiguous classifications (`tf_score` < 0.3 or > 0.55).
  * Created **pseudo-labeled T samples** from uncertain F cases.
  * Applied **upsampling** with:

    ```python
    from sklearn.utils import resample
    df_minority_upsampled = resample(...)
    ```

---

## 📊 Classification Model

Used a **TF-IDF Vectorizer**:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
```

Trained a simple **Logistic Regression classifier**:

```python
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_vec, y_train)
```

---

## 🧪 Evaluation Results

| Metric    | Class F | Class T |
| --------- | ------- | ------- |
| Precision | 0.79    | 0.79    |
| Recall    | 0.80    | 0.78    |
| F1-score  | 0.80    | 0.78    |

Overall **accuracy**: **79%** — Balanced across both classes.

---

## 📦 Final Dataset Preparation

* Selected top 300 most confident predictions per class (based on predicted probability).
* Created a **balanced dataset** of 600 samples.
* Constructed **OpenAI-compatible JSONL** format for fine-tuning:

```json
{
  "messages": [
    {"role": "system", "content": "You are a mental health assistant. Use emotional support and validation in your response."},
    {"role": "user", "content": "I'm feeling overwhelmed with work lately."},
    {"role": "assistant", "content": "It’s okay to feel overwhelmed sometimes. Make sure you’re taking care of yourself and resting when needed."}
  ]
}
```

The **system prompt** dynamically switches based on whether the response is T (logical) or F (emotional).

---

## ✅ Summary of Contributions

This project demonstrates:

* Real-world LLM adaptation through **data-driven personalization**
* **Zero-shot classification** as a weak supervision technique
* **Class balancing** via statistical thresholds + resampling
* Fine-tuning dataset preparation aligned with **OpenAI’s fine-tuning API**

---

## 🏗 Technologies & Libraries Used

* **Python**
* **Hugging Face Transformers**
* **Scikit-learn**
* **Pandas**
* **NLTK**
* **OpenAI Fine-tuning API**

---

## 💬 What I Learned

* Managing **class imbalance** in real-world, unstructured NLP datasets
* Leveraging **zero-shot models** for rapid labeling without manual annotation
* Preparing fine-tuning data with **high-confidence subsets**
* Understanding the nuances of balancing **logical vs. emotional** AI responses

## 📈 Future Work

* Integrate with a live chatbot framework (custom Django backend)
* Expand dataset to cover more nuanced mental health topics
* Implement multi-turn conversation flows and context retention


## 🌟 How to Use This Repository

1. Clone the repo:

   ```bash
   git clone https://github.com/yourusername/mental-health-chatbot.git
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run preprocessing:

   ```bash
   python ImprovedLLM.ipynb
   ```

---

If you like this project, feel free to ⭐️ star the repo or fork it for your own experiments!
