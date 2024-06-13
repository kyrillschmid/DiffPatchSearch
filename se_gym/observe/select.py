import abc
import rank_bm25
import logging
from . import read
from se_gym import api
import typing

__all__ = ["Selector", "BM25Selector", "FullSelector"]

logger = logging.getLogger("observe.select")


class Selector(abc.ABC):
    """
    The selector is the second element of the retrieval chain.
    For a given state, it selects the most relevant documents.
    """

    def __init__(self):
        self.cache = ""

    @abc.abstractmethod
    def _call_safe(
        self, state: api.State, documents: typing.List[read.Document]
    ) -> str: ...

    def clear_cache(self):
        self.cache = ""

    def __call__(
        self,
        state: typing.Union[api.State, typing.List[api.State]],
        documents: typing.List[read.Document],
    ) -> str:
        if isinstance(state, list):
            if len(state) == 0:
                raise ValueError("Population is empty. Nothing to observe.")
            return self._call_safe(state[0], documents)
        return self._call_safe(state, documents)

    def _call_safe_cached(
        self, state: api.State, documents: typing.List[read.Document]
    ):
        if self.cache:
            return self.cache
        self.cache = self._call_safe(state, documents)
        return self.cache

    def get_failing_issues(
        self, state: api.State, documents: typing.List[read.Document]
    ):
        docs = []
        for doc in documents:
            if state.fail_to_pass:
                if doc.path in state.fail_to_pass:
                    docs.append(doc.get_formatted())
        return "\n".join(docs)


class BM25Selector(Selector):
    """
    Select the most relevant document using Okapi BM25.
    """

    def __init__(self, tokenize=lambda x: x.split(" "), num_relevant_files=5):
        super().__init__()
        self.tokenize = tokenize
        self.num_relevant_files = num_relevant_files

    def _call_safe(
        self, state: api.State, documents: typing.List[read.Document]
    ) -> str:
        tokenized_documents = []
        for doc in documents:
            tokenized_documents.append(self.tokenize(doc.text))
        bm25 = rank_bm25.BM25Okapi(corpus=tokenized_documents)
        selected = bm25.get_top_n(
            state.issue.split(" "),
            documents,
            n=self.num_relevant_files,
        )
        return "\n".join(
            selected.get_formatted() for selected in selected
        ) + self.get_failing_issues(state, documents)


class FullSelector(Selector):
    """
    Select all documents. Only useful for the OracleReader or when context length is not important.
    """

    def _call_safe(
        self, state: api.State, documents: typing.List[read.Document]
    ) -> str:
        formatted = "".join([d.get_formatted() for d in documents])
        return formatted + self.get_failing_issues(state, documents)


class VectorSimilaritySelector(Selector): ...


class LLMSelector(Selector): ...
