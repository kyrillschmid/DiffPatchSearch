import typing
from . import select
from . import read
from . import compress
from se_gym import api


class Observer:
    def __init__(
        self,
        reader: typing.Optional[read.Reader] = None,
        selector: typing.Optional[select.Selector] = None,
        compressor: typing.Optional[compress.Compression] = None,
    ):
        if reader is None:
            reader = read.RawReader(root_dir=".")
        if selector is None:
            selector = select.BM25Selector()
        if compressor is None:
            compressor = compress.NoCompression()
        self.reader = reader
        self.selector = selector
        self.compressor = compressor

    def __call__(self, state):
        if isinstance(state, list):
            return [self(s) for s in state]
        all_documents = self.reader.get_documents()
        selected = self.selector(state, all_documents)
        full = self._get_issue(state) + selected + self._get_logs(state)
        return self.compressor(full)

    def _get_issue(self, state: api.State):
        return f"""\n\n
=========================
<ISSUE DESCRIPTION>
=========================
\n{state.issue}\n
=========================
</ISSUE DESCRIPTION>
=========================
"""

    def _get_logs(self, state: api.State):
        if not state.logs:
            return ""
        return f"""\n\n
=========================
<LOGS>
=========================
\n{state.logs}\n
=========================
</LOGS>
=========================
    """

    def from_env(self, env):
        self.reader = self.reader.from_env(env)
        self.clear_cache()
        return self

    def clear_cache(self):
        self.reader.clear_cache()
        self.selector.clear_cache()
        return self
    
