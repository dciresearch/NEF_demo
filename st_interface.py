from collections import Counter
import pandas as pd
import streamlit as st
import numpy as np
from src.extractors import get_processor_list, alias_to_class
from src.utils import LanguageDetector, get_supported_languages
from annotated_text import annotated_text


@st.cache(allow_output_mutation=True)
def load_language_detector():
    return LanguageDetector()


@st.cache(allow_output_mutation=True)
def load_processor(alias, lang):
    return alias_to_class(alias)(lang)


st.set_page_config(
    page_title="Extractor", layout="wide"
)
c30, c31, c32 = st.columns([2.5, 1, 3])

with c30:
    st.title("Fact Extractor")
    st.header("")

lang_detector = load_language_detector()
file = st.file_uploader("Загрузка текстовых файлов")

doc = None
if file:
    doc = file.read().decode("utf-8")

    languages = lang_detector.detect(doc)[0]

    major_language = Counter(languages).most_common(1)[0][0]

    if major_language not in get_supported_languages():
        st.warning("Language {} is not supported".format(major_language))
        st.stop()

if not file:
    st.markdown("## Пожалуйста, загрузите файл")
    st.stop()


ce, c1, ce, c2, ce = st.columns(
    [0.07, 2, 0.07, 5, 0.07])

possible_processors = get_processor_list(major_language)

with c1:
    processor_type = st.selectbox("Processor",
                                  options=possible_processors
                                  )
with c2:
    processor = load_processor(processor_type, major_language)
    triples = processor.get_relations(doc)
    df = pd.DataFrame(triples, columns=["Объект", "Отношение", "Субъект"])

    annotated_text(*processor.get_tokens_for_display(doc))
    st.markdown("#### Извлеченные факты:")
    st.table(df)
