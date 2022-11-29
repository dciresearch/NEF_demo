from collections import Counter
import pandas as pd
import streamlit as st
import numpy as np
from src.extractors import get_processor_list, alias_to_class
from annotated_text import annotated_text
from fast_api import get_lang, get_parsing

st.set_page_config(
    page_title="Extractor", layout="wide"
)
c30, c31, c32 = st.columns([2.5, 1, 3])

with c30:
    st.title("Fact Extractor")
    st.header("")

file = st.file_uploader("Загрузка текстовых файлов")

doc = None
if file:
    doc = file.read().decode("utf-8")
    major_language, supported = get_lang(doc)
    if not supported:
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
    error, processed = get_parsing(doc, processor_type)
    if error:
        st.warning(processed)
        st.stop()
    triples = processed["facts"]
    ents = ((e, c, s) for (e, c), s in processed["ents"])
    text_label = [tok if isinstance(tok, str) else tuple(
        tok) for tok in processed["text_labels"]]
    fact_df = pd.DataFrame(triples, columns=["Объект", "Отношение", "Субъект"])
    ents_df = pd.DataFrame(ents, columns=["Сущность", "Частота", "Источник"])
    annotated_text(*text_label)
    tab1, tab2 = st.tabs(["Сущности", "Факты"])
    with tab1:
        tab1.subheader("Извлеченные Сущности:")
        st.dataframe(ents_df, use_container_width=True)
    with tab2:
        tab2.subheader("Извлеченные факты:")
        st.dataframe(fact_df, use_container_width=True)
