# CognitiveML Suite

 An interactive web-based dashboard designed to bridge the gap between complex artificial intelligence algorithms and intuitive user business applications. The system processes text, images, speech, and transactional tabular databases to deliver actionable business intelligence and automated cognitive responses.

An all-in-one web application dashboard integrating classic unsupervised machine learning algorithms with state-of-the-art Deep Learning and Natural Language Processing (NLP) models. The platform enables users to perform complex data clustering, computer vision classification, speech transcription, text-to-speech generation, machine translation, named entity recognition, and transactional association mining through a unified and interactive web interface.

---

## Key Features & Capabilities

### 1. Unsupervised Machine Learning (Clustering Suite)
*   **K-Means Clustering**: Custom tabular CSV parsing, missing value imputation via `SimpleImputer` (mean strategy), and visual clustering using `scikit-learn`'s KMeans algorithm. Outputs 2D scatter plots mapping clusters and centroids dynamically.
*   **DBSCAN Clustering**: Density-based spatial clustering of applications with noise. Features automated feature standard scaling (`StandardScaler`), epsilon (`eps`), and minimum sample size configuration to identify core groups and eliminate outliers.

### 2. Deep Learning & Computer Vision
*   **CNN Gender Classification**: Image classification utilizing a fine-tuned ResNet-based CNN (`rizvandwiki/gender-classification-2`) via Hugging Face pipelines to classify gender from user-uploaded images with confidence percentages.

### 3. Advanced Natural Language Processing (NLP)
*   **Automatic Speech Recognition (ASR)**: Uses OpenAI's `Whisper-Tiny` model to transcribe uploaded audio files (e.g., WAV) into high-fidelity text.
*   **Generative Sentiment Analysis**: Pipelines transcribed text from the Whisper ASR model into a sequence-to-sequence large language model (`google/flan-t5-small`) to evaluate and classify positive or negative emotional polarity.
*   **Voice Q&A (Voice In, Voice Out)**: An interactive voice assistant that transcribes speech input via Whisper, generates a context-aware answer via FLAN-T5, and converts the text answer into synthesized speech using Google Text-to-Speech (gTTS).
*   **Named Entity Recognition (NER)**: Sequence token classification using a fine-tuned DistilBERT model (`elastic/distilbert-base-cased-finetuned-conll03-english`) to identify and catalog Persons, Locations, Organizations, and Miscellaneous entities.
*   **Machine Translation (English-to-Urdu)**: Neural machine translation leveraging Helsinki-NLP's MarianMT transformer sequence-to-sequence model (`Helsinki-NLP/opus-mt-en-ur`).

### 4. Market Basket Analysis (Association Rules)
*   **Apriori Algorithm**: Transactional database processing using `mlxtend`s TransactionEncoder and Apriori algorithm. Computes market basket rules based on custom user-defined minimum support, confidence, and lift thresholds.

---

## System Architecture & Technology Stack

*   **Backend Framework**: Flask (Python 3) with threaded background model pre-loading
*   **Data Analysis & Imputation**: Pandas, NumPy, Scikit-Learn
*   **Deep Learning & LLMs**: PyTorch, Hugging Face Transformers (Whisper, MarianMT, DistilBERT, FLAN-T5)
*   **Visualization**: Matplotlib (Agg backend for headless server execution)
*   **Association Mining**: Mlxtend (Apriori, association_rules)
*   **Text-To-Speech (TTS)**: gTTS
*   **Frontend**: HTML5, Vanilla CSS, Jinja2 Templates

---

## Repository Structure

```
├── app.py                      # Main Flask application & routing coordinator
├── data inputs/                 # Sample datasets (CSV, WAV, JPG) for quick testing
│   ├── female.jpg
│   ├── male.jpg
│   ├── harvard.wav
│   ├── marketBasket.csv
│   └── student_study_data.csv
├── model/                      # Pickled models and serialization files
│   ├── imputer.pkl
│   └── model.pkl
├── static/                     # Directory for generated audio files (e.g. answer.mp3)
├── templates/                  # Jinja2 HTML layout files
│   ├── dashboard.html          # Unified dashboard interface
│   ├── kmeans.html             # K-means configuration view
│   ├── kmeans_result.html      # K-means visualization page
│   ├── dbscan.html             # DBSCAN configuration view
│   ├── dbscan_result.html      # DBSCAN visualization page
│   ├── cnnGender.html          # Gender classification interface
│   ├── voiceSentiment.html     # Voice sentiment analysis UI
│   ├── voiceQa.html            # Voice assistant UI
│   ├── textQa.html             # Text QA interface
│   ├── translate.html          # English-to-Urdu translation UI
│   ├── ner.html                # Named Entity Recognition UI
│   └── apriori.html            # Association rules table UI
├── uploads/                    # Temporary storage for uploaded media files
└── requirements.txt            # Project dependencies list
```

---

## Installation & Setup

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/ukasha167/CognitiveML-Suite.git
    cd CognitiveML-Suite
    ```

2.  **Create a Virtual Environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Required Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *If `requirements.txt` is not present, install manually:*
    ```bash
    pip install flask pandas matplotlib mlxtend scikit-learn transformers pillow gtts torch sentencepiece protobuf soundfile
    ```

4.  **Run the Application**:
    ```bash
    python app.py
    ```
    The application will automatically download the pre-trained Hugging Face weights on its first boot and launch a local web server at `http://127.0.0.1:5000/`.

---

## Usage Guide

*   **Clustering**: Upload any numerical CSV. The app will extract the first two numerical features, clean missing data, normalize features (for DBSCAN), and plot the clustering decision boundaries.
*   **Vision**: Upload a portrait image (PNG/JPG) to test the CNN gender classifier.
*   **Voice/NLP**: Upload a WAV file for transcription and sentiment analysis. Or record/upload audio questions to test the voice-interactive chat bot.
*   **Market Basket Analysis**: Upload a transactional CSV where each row is a comma-separated list of items purchased together. Adjust minimum support and confidence thresholds to generate rules.
