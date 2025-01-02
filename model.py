# model.py
from lib import *
import re
import joblib

# Load model CRF yang sudah dilatih
model = joblib.load('nercrf_model.pkl')

# Fungsi untuk ekstraksi fitur dari setiap kata dalam kalimat


def word2features(sent, i):
    word = sent[i]
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
    }
    if i > 0:
        word1 = sent[i - 1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
        })
    else:
        features['BOS'] = True  # Awal kalimat

    if i < len(sent) - 1:
        word1 = sent[i + 1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
        })
    else:
        features['EOS'] = True  # Akhir kalimat

    return features

# Ekstraksi fitur dari seluruh kalimat


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

# Fungsi rule-based tagging yang dikombinasikan dengan model CRF


def rule_based_tagging(words, model_prediction):
    tags = []
    for i, (word, pred) in enumerate(zip(words, model_prediction)):
        if pred == 'O' and word.istitle():
            tags.append("B-PER" if i == 0 or tags[-1] == "O" else "I-PER")
        else:
            tags.append(pred)
    return tags

# Fungsi untuk prediksi NER dari input teks


def predict_text(input_text):
    # Hapus tanda baca dari kalimat
    input_text = re.sub(r'[.,!?;]', '', input_text)

    # Pisahkan kalimat menjadi kata-kata
    words = input_text.split()

    # Ekstraksi fitur dari input
    features = sent2features(words)

    if model is None:
        print("Model belum dimuat.")
        return None

    # Prediksi model CRF
    try:
        prediction = model.predict([features])[0]
    except Exception as e:
        print(f"Error saat prediksi: {e}")
        return None

    # Gabungkan hasil prediksi model dan rule-based
    final_prediction = rule_based_tagging(words, prediction)

    # Tampilkan hasil prediksi
    result = list(zip(words, final_prediction))
    return result

# Fungsi untuk mewarnai tag NER


def style_tag(word, tag):
    colors = {
        'B-PER': '#FF6666',
        'I-PER': '#FF6666',
        'B-ADJ': '#66FF66',
        'I-ADJ': '#66FF66',
        'B-ANM': '#FFFF66',
        'I-ANM': '#FFFF66',
        'B-GODS': '#6666FF',
        'I-GODS': '#6666FF',
        'B-OBJ': '#CC66FF',
        'I-OBJ': '#CC66FF',
        'O': '#D2B48C'
    }
    color = colors.get(tag, '#FFFFFF')
    return f'<span style="background-color: {color}; padding: 2px 4px; border-radius: 3px; margin-right: 4px;">{word} ({tag})</span>'
