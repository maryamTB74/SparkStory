"""Keyword search, in pure Python.

No dependency: BM25 over a hundred chunks is a dozen lines of arithmetic, and
``rank_bm25`` would be a package to audit and pin for no gain at this size.

Why keyword search at all, when there are embeddings: lesson 9 is explicit that
BM25 wins on exact terms, and this corpus is *made* of exact terms. "The Moon has
no air" is a bag of specific words, and a query mentioning the moon should not
depend on a 256-dimensional approximation to find it. The two retrievers fail
differently, which is the argument for running both.

Formula is the standard Okapi BM25 with the usual constants. They are not tuned,
because tuning them without the labelled eval set would be guessing -- and once
that set exists, hit-rate@3 makes tuning measurable instead.
"""

import math
import re
from collections import Counter

import numpy as np

#: Term-frequency saturation. Above this, repeating a word stops helping much.
_K1 = 1.5
#: Length normalisation. 0 ignores document length, 1 fully normalises by it.
_B = 0.75

#: Function words, dropped from both documents and queries.
#:
#: IDF is not enough on its own, and the first build of the real index is what
#: showed why. The query "could a flag wave on the moon?" ranked the chunk about
#: *sound* first, because "could" is rare in a corpus of short factual statements
#: and therefore earned a *high* idf -- while the chunk it matched only contained
#: "could not hear". A word can be common in English and rare in the corpus, and
#: for those words idf points exactly the wrong way.
#:
#: Kept short and hand-picked rather than imported: a full stoplist would need a
#: dependency, and an aggressive one would strip "no", which is load-bearing in
#: half these facts ("the Moon has *no* air").
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "this",
        "that",
        "these",
        "those",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "am",
        "do",
        "does",
        "did",
        "doing",
        "done",
        "can",
        "could",
        "will",
        "would",
        "shall",
        "should",
        "may",
        "might",
        "must",
        "have",
        "has",
        "had",
        "having",
        "i",
        "you",
        "he",
        "she",
        "it",
        "we",
        "they",
        "them",
        "him",
        "her",
        "its",
        "their",
        "our",
        "my",
        "your",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "from",
        "with",
        "by",
        "about",
        "into",
        "over",
        "under",
        "and",
        "or",
        "but",
        "if",
        "then",
        "than",
        "so",
        "as",
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "when",
        "where",
        "why",
        "how",
        "there",
        "here",
    ]
)


def tokenize(text: str) -> list[str]:
    """Split text into lowercase word tokens.

    Deliberately the same rule ``FakeEmbedder`` uses. Both halves of a hybrid
    search have to agree on what a word is, or a comma decides which retriever can
    see a chunk -- and that kind of disagreement is invisible until a specific
    query misses.
    """
    return re.findall(r"\w+", text.lower())


def _content_words(text: str) -> list[str]:
    """Tokens with function words removed. Used for both documents and queries.

    Separate from ``tokenize`` so the tokenizer stays a pure word-splitter shared
    with the embedder -- stopword removal is a BM25 decision, and the vector half
    should still see the whole sentence.
    """
    return [word for word in tokenize(text) if word not in _STOPWORDS]


class BM25Index:
    """Okapi BM25 over a fixed list of documents.

    Built once from the store's chunks and held for the process. Rebuilding per
    query would re-tokenise the whole corpus on the cheapest path in the system.
    """

    def __init__(self, documents: list[str]) -> None:
        self._tokens = [_content_words(document) for document in documents]
        self._lengths = np.array(
            [len(tokens) for tokens in self._tokens], dtype=np.float32
        )
        self._average_length = (
            float(self._lengths.mean()) if len(self._lengths) else 0.0
        )
        self._frequencies = [Counter(tokens) for tokens in self._tokens]

        document_count = len(documents)
        appearances: Counter[str] = Counter()
        for frequency in self._frequencies:
            appearances.update(frequency.keys())

        # The standard BM25 idf, which goes slightly negative for a term in more
        # than half the corpus. `1 +` inside the log keeps it positive instead: a
        # very common term should contribute nothing, not actively penalise a
        # document for containing it.
        self._idf = {
            term: math.log(1 + (document_count - count + 0.5) / (count + 0.5))
            for term, count in appearances.items()
        }

    def matched_terms(self, query: str) -> int:
        """How many distinct query terms exist in the corpus vocabulary at all.

        Used to decide whether this retriever has an opinion worth fusing. With a
        single generic term -- "moon", against a corpus where most chunks mention
        the Moon -- every document matches and BM25 ranks them by *length*, which
        is noise rather than relevance. Fusing that noise costs a real hit: the
        labelled set caught exactly one, where "why would you bounce if you walked
        on the moon?" contained no corpus term except "moon" (this corpus says
        "jump far higher", not "bounce") and keyword ranking pushed a shorter
        moon chunk over the right one.
        """
        return len({term for term in _content_words(query) if term in self._idf})

    def scores(self, query: str) -> np.ndarray:
        """Score every document against ``query``. Higher is better.

        Returns all-zeros when no query term appears anywhere, which is a normal
        outcome rather than a failure -- a premise about a lost teddy shares no
        term with a corpus of space facts.
        """
        scores = np.zeros(len(self._tokens), dtype=np.float32)
        if not len(self._lengths):
            return scores

        for term in _content_words(query):
            idf = self._idf.get(term)
            if idf is None:
                continue
            for index, frequency in enumerate(self._frequencies):
                count = frequency.get(term, 0)
                if not count:
                    continue
                length_penalty = (
                    1 - _B + _B * self._lengths[index] / self._average_length
                )
                scores[index] += (
                    idf * (count * (_K1 + 1)) / (count + _K1 * length_penalty)
                )
        return scores
