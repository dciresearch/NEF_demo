from itertools import zip_longest, chain
import json
import toml
import requests
import uvicorn
from fastapi import FastAPI
from collections import Counter
from src.extractors import get_processor_list, alias_to_class
from src.utils import LanguageDetector, get_supported_languages
from functools import lru_cache
from pydantic import BaseModel

FASTAPI_CONFIG = toml.load(".streamlit/config.toml")["fastapi"]
FASTAPI_ADDRESS = "{}:{}".format(
    FASTAPI_CONFIG["fastapiAddress"], FASTAPI_CONFIG["fastapiPort"])


def load_language_detector():
    return LanguageDetector()


@ lru_cache(maxsize=2)
def load_processor(alias, lang):
    return alias_to_class(alias)(lang)


lang_detector = load_language_detector()

app = FastAPI()


class ParseDoc(BaseModel):
    doc: str
    processor_type: str = None
    return_text_labels: bool = False


@app.post("/list_processors")
def list_processors(lang: str = "ru"):
    return get_processor_list(lang)


@ app.post("/detect_language")
def get_language(input: ParseDoc):
    languages = lang_detector.detect(input.doc)[0]
    major_language = Counter(languages).most_common(1)[0][0]
    supported = major_language in get_supported_languages()
    return major_language, supported


@ app.post("/parse_document")
def parse_document(input: ParseDoc):
    major_language, supported = get_language(input)
    if not supported:
        return (-1, "Language {} is not supported".format(major_language))
    possible_processors = set(get_processor_list(major_language))
    if input.processor_type not in possible_processors:
        return (-1, "Processor {} not supported for language {}".format(
            input.processor_type, major_language))
    processor = load_processor(input.processor_type, major_language)
    triples, phrase_ents = processor.get_relations(input.doc)
    text_label, true_ents = processor.get_tokens_for_display(input.doc)
    true_ents_set = set(true_ents)
    true_ents = sorted(Counter(true_ents).items(), key=lambda x: -x[1])
    phrase_ents = sorted(Counter(
        e for e in phrase_ents if e not in true_ents_set).items(), key=lambda x: -x[1])
    ents = list(chain(zip_longest(true_ents, [], fillvalue="NER"), zip_longest(
        phrase_ents, [], fillvalue="NounPhrases")))
    res = {"facts": triples, "ents": ents}
    if input.return_text_labels:
        res["text_labels"] = text_label
    return (0, res)


def get_lang(text):
    data = json.dumps({"doc": text})
    return requests.post("http://{}/{}".format(FASTAPI_ADDRESS, "detect_language"), data).json()


def get_parsing(text, processor_type):
    data = json.dumps({"doc": text, "processor_type": processor_type,
                       "return_text_labels": True})
    return requests.post("http://{}/{}".format(FASTAPI_ADDRESS, "parse_document"), data).json()


if __name__ == "__main__":
    uvicorn.run(
        app, host=FASTAPI_CONFIG["fastapiAddress"], port=FASTAPI_CONFIG["fastapiPort"])
