import os
import csv
import io
import base64
import threading

from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.utils import secure_filename

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from PIL import Image
from gtts import gTTS
import torch

app = Flask(__name__)
app.secret_key = "IDK"

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("static", exist_ok=True)

_models = {}
_loading_locks = {}
_global_lock = threading.Lock()

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")

def get_model(name):
    with _global_lock:
        if name not in _loading_locks:
            _loading_locks[name] = threading.Lock()
    
    with _loading_locks[name]:
        if name in _models:
            return _models[name]
        
        device = get_device()
        device_arg = 0 if device.type == "cuda" else device

        try:
            if name == "gender_classifier":
                _models[name] = pipeline("image-classification", model="rizvandwiki/gender-classification-2", device=device_arg)
            elif name == "asr_pipeline":
                _models[name] = pipeline("automatic-speech-recognition", model="openai/whisper-tiny", device=device_arg)
            elif name == "qa_tokenizer":
                _models[name] = AutoTokenizer.from_pretrained("google/flan-t5-small")
            elif name == "qa_model":
                _models[name] = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small").to(device)
            elif name == "translation_tokenizer":
                _models[name] = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ur")
            elif name == "translation_model":
                _models[name] = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-ur").to(device)
            elif name == "ner_pipeline":
                _models[name] = pipeline("token-classification", model="elastic/distilbert-base-cased-finetuned-conll03-english", aggregation_strategy="simple", device=device_arg)
            
            return _models[name]
        except Exception as init_error:
            raise init_error

def preload_model(name):
    def task():
        try:
            get_model(name)
        except Exception:
            pass
    threading.Thread(target=task, daemon=True).start()

def get_valid_text(field_name, error_message):
    text = request.form.get(field_name)
    if not text or text.strip() == "":
        return None, error_message
    return text.strip(), None

def process_numeric_csv(file):
    df = pd.read_csv(file)
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.shape[1] < 2:
        raise ValueError("CSV must contain at least 2 numerical columns.")
    numeric_df = numeric_df.iloc[:, :2]
    imputer = SimpleImputer(strategy="mean")
    X_clean = imputer.fit_transform(numeric_df)
    return X_clean

def plot_to_base64():
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    plt.close()
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf8')

def generate_answer(prompt, max_length=50):
    qa_tokenizer = get_model("qa_tokenizer")
    qa_model = get_model("qa_model")
    device = get_device()
    inputs = qa_tokenizer(prompt, return_tensors="pt").to(device)
    outputs = qa_model.generate(**inputs, max_length=max_length, do_sample=True, temperature=0.7, top_p=0.9, repetition_penalty=1.1)
    return qa_tokenizer.decode(outputs[0], skip_special_tokens=True)

@app.route("/")
def dashboard():
    return render_template("dashboard.html")

@app.route("/kmeans", methods=["GET", "POST"])
def kmeans():
    error = None
    if request.method == "POST":
        if "file" not in request.files or request.files["file"].filename == "":
            error = "No file uploaded."
        else:
            try:
                k = int(request.form.get("k_clusters", 3))
                X_clean = process_numeric_csv(request.files["file"])

                kmeans_model = KMeans(n_clusters=k, random_state=42)
                cluster_labels = kmeans_model.fit_predict(X_clean)

                plt.figure(figsize=(6, 4))
                scatter = plt.scatter(X_clean[:, 0], X_clean[:, 1], c=cluster_labels, cmap="viridis")
                plt.xlabel("Feature 1")
                plt.ylabel("Feature 2")
                plt.title(f"K-Means Clustering (k={k})")
                plt.colorbar(scatter, label="Cluster Group")

                plot_url = plot_to_base64()
                return render_template("kmeans_result.html", k=k, plot_url=plot_url)
            except Exception as e:
                error = str(e)

    return render_template("kmeans.html", error=error)

@app.route("/dbscan", methods=["GET", "POST"])
def dbscan():
    error = None
    if request.method == "POST":
        if "file" not in request.files or request.files["file"].filename == "":
            error = "No file uploaded."
        else:
            try:
                eps_value = float(request.form.get("eps", 0.5))
                min_samples_value = int(request.form.get("min_samples", 5))
                X_clean = process_numeric_csv(request.files["file"])

                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_clean)

                dbscan_model = DBSCAN(eps=eps_value, min_samples=min_samples_value)
                cluster_labels = dbscan_model.fit_predict(X_scaled)

                n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                n_noise = list(cluster_labels).count(-1)

                plt.figure(figsize=(6, 4))
                scatter = plt.scatter(X_clean[:, 0], X_clean[:, 1], c=cluster_labels, cmap="Spectral")
                plt.xlabel("Feature 1")
                plt.ylabel("Feature 2")
                plt.title(f"DBSCAN (eps={eps_value}, min_samples={min_samples_value})")
                plt.colorbar(scatter, label="Cluster Group (-1 is Noise)")

                plot_url = plot_to_base64()
                return render_template(
                    "dbscan_result.html",
                    eps=eps_value,
                    min_samples=min_samples_value,
                    n_clusters=n_clusters,
                    n_noise=n_noise,
                    plot_url=plot_url
                )
            except Exception as e:
                error = str(e)

    return render_template("dbscan.html", error=error)

@app.route("/cnnGender", methods=["GET", "POST"])
def cnnGender():
    label = None
    confidence = None
    error = None

    if request.method == "POST":
        if "image" not in request.files or request.files["image"].filename == "":
            error = "No image uploaded"
        else:
            try:
                gender_classifier = get_model("gender_classifier")
                file = request.files["image"]
                image = Image.open(file.stream).convert("RGB")
                results = gender_classifier(image)
                best_prediction = results[0]

                label = best_prediction["label"].upper()
                confidence = f"{best_prediction['score'] * 100:.2f}%"
            except Exception as e:
                error = str(e)
    else:
        preload_model("gender_classifier")

    return render_template("cnnGender.html", label=label, confidence=confidence, error=error)

@app.route("/voiceSentiment", methods=["GET", "POST"])
def voiceSentiment():
    transcribed_text = None
    sentiment = None
    confidence = None
    error = None

    if request.method == "POST":
        if "audio" not in request.files or request.files["audio"].filename == "":
            error = "No audio uploaded"
        else:
            filepath = None
            try:
                asr_pipeline = get_model("asr_pipeline")
                file = request.files["audio"]
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(filepath)

                transcription_result = asr_pipeline(filepath)
                transcribed_text = transcription_result["text"]

                prompt = f"Classify the sentiment of the following text as either Positive or Negative: '{transcribed_text}'"
                sentiment_result = generate_answer(prompt, max_length=10)

                sentiment = sentiment_result.strip().capitalize()
                confidence = "N/A (Generative Model)"
            except Exception as e:
                error = str(e)
            finally:
                if filepath and os.path.exists(filepath):
                    os.remove(filepath)
    else:
        preload_model("asr_pipeline")
        preload_model("qa_tokenizer")
        preload_model("qa_model")

    return render_template(
        "voiceSentiment.html",
        transcribed_text=transcribed_text,
        sentiment=sentiment,
        confidence=confidence,
        error=error,
    )

@app.route("/voiceQa", methods=["GET", "POST"])
def voiceQa():
    question_text = None
    answer_text = None
    audio_path = None
    error = None

    if request.method == "POST":
        if "audio" not in request.files or request.files["audio"].filename == "":
            error = "No audio uploaded"
        else:
            input_filepath = None
            try:
                asr_pipeline = get_model("asr_pipeline")
                file = request.files["audio"]
                filename = secure_filename(file.filename)
                input_filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(input_filepath)

                question_text = asr_pipeline(input_filepath)["text"]
                answer_text = generate_answer(question_text, max_length=50)

                tts = gTTS(text=answer_text, lang="en")
                output_filename = "answer.mp3"
                output_filepath = os.path.join("static", output_filename)
                tts.save(output_filepath)

                audio_path = output_filename
            except Exception as e:
                error = str(e)
            finally:
                if input_filepath and os.path.exists(input_filepath):
                    os.remove(input_filepath)
    else:
        preload_model("asr_pipeline")
        preload_model("qa_tokenizer")
        preload_model("qa_model")

    return render_template(
        "voiceQa.html",
        question=question_text,
        answer=answer_text,
        audio_path=audio_path,
        error=error,
    )

@app.route("/textQa", methods=["GET", "POST"])
def textQa():
    question = None
    answer = None
    error = None

    if request.method == "POST":
        question, error = get_valid_text("question", "Please type a question.")
        if not error:
            try:
                answer = generate_answer(question, max_length=150)
            except Exception as e:
                error = str(e)
    else:
        preload_model("qa_tokenizer")
        preload_model("qa_model")

    return render_template("textQa.html", question=question, answer=answer, error=error)

@app.route("/translate", methods=["GET", "POST"])
def translate():
    english_text = None
    urdu_text = None
    error = None

    if request.method == "POST":
        english_text, error = get_valid_text("english_text", "Please enter some text to translate.")
        if not error:
            try:
                translation_tokenizer = get_model("translation_tokenizer")
                translation_model = get_model("translation_model")
                device = get_device()
                
                inputs = translation_tokenizer(english_text, return_tensors="pt").to(device)
                outputs = translation_model.generate(**inputs, max_length=150)
                urdu_text = translation_tokenizer.decode(outputs[0], skip_special_tokens=True)
            except Exception as e:
                error = str(e)
    else:
        preload_model("translation_tokenizer")
        preload_model("translation_model")

    return render_template("translate.html", english_text=english_text, urdu_text=urdu_text, error=error)

@app.route("/ner", methods=["GET", "POST"])
def ner():
    input_text = None
    entities = []
    error = None

    if request.method == "POST":
        input_text, error = get_valid_text("input_text", "Please enter some text to analyze.")
        if not error:
            try:
                ner_pipeline = get_model("ner_pipeline")
                raw_entities = ner_pipeline(input_text)
                label_map = {
                    "PER": "Person",
                    "LOC": "Location",
                    "ORG": "Organization",
                    "MISC": "Miscellaneous",
                }

                for ent in raw_entities:
                    readable_label = label_map.get(ent["entity_group"], ent["entity_group"])
                    entities.append({
                        "word": ent["word"],
                        "label": readable_label,
                        "score": f"{ent['score'] * 100:.1f}%",
                    })
            except Exception as e:
                error = str(e)
    else:
        preload_model("ner_pipeline")

    return render_template("ner.html", input_text=input_text, entities=entities, error=error)

@app.route("/apriori", methods=["GET", "POST"])
def run_apriori():
    rules_html = None
    error = None

    if request.method == "POST":
        if "file" not in request.files or request.files["file"].filename == "":
            error = "No file uploaded."
        else:
            try:
                min_support = float(request.form.get("min_support", 0.1))
                min_confidence = float(request.form.get("min_confidence", 0.5))

                file = request.files["file"]
                
                stream = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
                reader = csv.reader(stream)
                transactions = []
                for row in reader:
                    cleaned_row = [item.strip() for item in row if item.strip()]
                    if cleaned_row:
                        transactions.append(cleaned_row)

                te = TransactionEncoder()
                te_ary = te.fit(transactions).transform(transactions)
                df = pd.DataFrame(te_ary, columns=te.columns_)

                frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)

                if frequent_itemsets.empty:
                    error = f"No itemsets found with at least {min_support*100}% support. Try lowering the Min Support."
                else:
                    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

                    if rules.empty:
                        error = f"No rules found with at least {min_confidence*100}% confidence. Try lowering the Min Confidence."
                    else:
                        rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(list(x)))
                        rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(list(x)))

                        display_rules = rules[["antecedents", "consequents", "support", "confidence", "lift"]].copy()
                        display_rules["support"] = (display_rules["support"] * 100).round(2).astype(str) + "%"
                        display_rules["confidence"] = (display_rules["confidence"] * 100).round(2).astype(str) + "%"
                        display_rules["lift"] = display_rules["lift"].round(2)

                        rules_html = display_rules.to_html(classes="data-table", index=False, justify="left")

            except Exception as e:
                error = str(e)

    return render_template("apriori.html", rules_html=rules_html, error=error)

if __name__ == "__main__":
    app.run(debug=True)
