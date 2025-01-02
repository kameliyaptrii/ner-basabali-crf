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

# Fungsi untuk prediksi NER dari input teks


def predict_text(input_text):
    # Pisahkan kalimat menjadi kata-kata
    words = input_text.split()
    print(f"Kata-kata: {words}")  # Debugging: cetak kata-kata

    # Ekstraksi fitur dari input
    features = sent2features(words)
    print(f"Fitur: {features}")  # Debugging: cetak fitur yang diekstraksi

    if model is None:
        print("Model belum dimuat.")  # Debugging jika model kosong
        return None

    # Lakukan prediksi menggunakan model CRF
    try:
        prediction = model.predict([features])[0]
        print(f"Prediksi: {prediction}")  # Debugging: cetak hasil prediksi
    except Exception as e:
        print(f"Error saat prediksi: {e}")
        return None

    # Tampilkan hasil prediksi
    result = list(zip(words, prediction))
    return result  # Kembalikan hasil prediksi

    # Output hasil
    st.write("Hasil Prediksi:")
    for word, tag in result:
        st.write(f"{word}: {tag}")