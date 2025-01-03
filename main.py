
import streamlit as st
from model import predict_text, style_tag


def main():
    st.title(":green[NER BASA BALI]")
    st.write("""
        **Named Entity Recognition (NER)** adalah sebuah teknik dalam pemrosesan bahasa alami (Natural Language Processing) 
        yang digunakan untuk mengekstraksi entitas tertentu dari sebuah teks. Model ini mendeteksi entitas seperti :blue[**nama orang, dewa, dan hewan**] dalam teks berbahasa Bali.
    """)

    teks = st.text_input('Masukkan Kalimat')

    if st.button('Prediksi Kalimat'):
        if teks:
            prediction = predict_text(teks)

            if prediction is not None:
                st.write("Hasil Prediksi NER:")
                styled_result = ''.join([style_tag(word, tag)
                                        for word, tag in prediction])
                st.markdown(styled_result, unsafe_allow_html=True)
            else:
                st.write("Prediksi NER gagal.")
        else:
            st.write("Masukkan teks terlebih dahulu.")


if __name__ == '__main__':
    main()
