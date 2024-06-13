import abc

__all__ = ["Compression", "NoCompression"]

class Compression(abc.ABC):
    @abc.abstractmethod
    def __call__(original_text: str) -> str:
        pass


class NoCompression(Compression):
    """
    No compression.
    """

    def __call__(self, original_text: str) -> str:
        return original_text


class SPRCompressor(Compression):
    """
    Use Sparse Priming Representations to compress big code segments.
    Based on https://medium.com/@dave-shap/beyond-vector-search-knowledge-management-with-generative-ai-6c2d10b481a0
    """
