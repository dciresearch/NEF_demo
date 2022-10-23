from itertools import chain
from functools import lru_cache
import re
import spacy
from spacy.cli.download import download as spacy_download
from spacy.util import is_package as package_check
import stanza
import logging

spacy_models = {
    "en": "en_core_web_lg",
    "ru": "ru_core_news_lg"
}


valid_processors = {
    "ru": ["Spacy"],
    "en": ["Spacy", "Stanza (OpenIE)"],
}


def get_processor_list(lang):
    return valid_processors[lang]


def get_compounds(el):
    compounds = [el]
    for ch in el.children:
        if ch.dep_ == 'compound':
            compounds.append(ch)
    return " ".join([t.text for t in sorted(compounds, key=lambda x: x.i)])


def get_verb(el, max_depth=2):
    if el.pos_ == 'VERB' or max_depth == 0:
        return [el]
    else:
        return [el] + get_verb(el.head, max_depth-1)


def get_subtree(el):
    def get_st(children):
        return [tr for ch in children for tr in get_subtree(ch) if ch.pos_ not in "VERB PUNCT"]
    if not el.children:
        return [el]
    else:
        return get_st(el.lefts)+[el]+get_st(el.rights)


def subtree_to_text(subtree):
    return " ".join((t.text for t in subtree))


def get_object(el, max_depth=6):
    if max_depth == 0 or not el or el.pos_ == 'VERB':
        return [""]
    if el.pos_ == "NOUN":
        subtree = get_subtree(el)
        return [subtree_to_text(subtree)]
    else:
        return [" ".join([el.text, st]) for ch in el.children for st in get_object(ch, max_depth-1) if st]


def get_subj_triples(ent):
    triples = []
    relation_path = get_verb(ent.root.head)
    relation = relation_path[-1]
    for child in relation.children:
        if (len(relation_path) > 1 and child.i == relation_path[-2].i) or child.i == ent.root.i:
            continue
        objects = get_object(child)
        for obj in objects:
            if not obj:
                continue
            triples.append(
                (ent.text, "{}".format(get_compounds(relation)), obj))
    return triples


def get_appos_triples(ent):
    triples = []
    relation = ent.root.head
    for ch in relation.children:
        if ch.dep_ == "nmod":
            triples.append((ent.text, "является", " ".join(
                (relation.text, subtree_to_text(ch.subtree)))))
    return triples


def extract_relations(doc):
    triples = []
    for ent in doc.ents:
        if "subj" in ent.root.dep_:
            triples.extend(get_subj_triples(ent))
        if "appos" in ent.root.dep_:
            triples.extend(get_appos_triples(ent))

    return triples


bad_lineend = re.compile(r"([\w])(\s*\n)")


def fix_lineend(text):
    return bad_lineend.sub(r"\1. ", text)


class SpacyExtractor:
    def __init__(self, lang='en'):
        model_name = spacy_models[lang]
        if not package_check(model_name):
            logging.warning(
                "Model {} not installed. Downloading the model...".format(model_name))
            spacy_download(model_name)
        self.nlp = spacy.load(model_name)

    @lru_cache(maxsize=1)
    def process_document(self, text):
        return self.nlp(fix_lineend(text))

    def get_relations(self, text):
        doc = self.process_document(text)
        triples = []
        for sent in doc.sents:
            triples.extend(extract_relations(sent))
        return set(triples)

    def get_entities(self, text):
        doc = self.process_document(text)
        return tuple((ent.text for ent in doc.ents))

    def get_tokens_for_display(self, text):
        doc = self.process_document(text)
        tokens = [tok.text_with_ws for tok in doc]
        # get ent spans
        ent_slices = [(ent.start, ent.end, ent.label_) for ent in doc.ents]
        # get gap token spans
        gaps = ((a[1], b[0]) for a, b in zip(ent_slices, ent_slices[1:]))
        # fill gaps
        full_slices = list(chain(*zip(ent_slices, gaps)))
        if full_slices[0][0] != 0:
            full_slices = [(0, full_slices[0][0])]+full_slices
        if full_slices[-1][1] != len(tokens):
            full_slices.append((full_slices[-1][1], len(tokens)))

        # finally get display token list
        span_generator = (tokens[slice(*sl)] if len(sl) == 2 else [
                          ("".join(tokens[slice(*sl[:2])]), sl[2])] for sl in full_slices)
        display_tokens = [tok for span in span_generator for tok in span]

        return display_tokens


class OpenIEExtractor:
    def __init__(self, lang=None):
        stanza.install_corenlp()
        self.client = stanza.server.CoreNLPClient(
            timeout=30000,
            memory='2G')

    @lru_cache(maxsize=1)
    def process_document(self, text):
        return self.client.annotate(fix_lineend(text))

    def __del__(self):
        self.client.__exit__()

    def get_relations(self, text):
        doc = self.process_document(text)
        triples = []
        for sent in doc.sentence:
            triples.extend(((tr.subject, tr.relation, tr.object)
                           for tr in sent.kbpTriple))
        return set(triples)

    def get_entities(self, text):
        doc = self.process_document(text)
        return tuple((ment.entityMentionText for ment in doc.mentions))

    def get_tokens_for_display(self, text):
        doc = self.process_document(text)
        tokens = [(tok.word + tok.after, tok.ner) if tok.ner != 'O'
                  else tok.word + tok.after for sent in doc.sentence for tok in sent.token]
        return tokens


processor_aliases = {
    "Spacy": SpacyExtractor,
    "Stanza (OpenIE)": OpenIEExtractor
}


def alias_to_class(alias):
    return processor_aliases[alias]