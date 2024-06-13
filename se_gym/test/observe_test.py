import os
import se_gym


dummy_file = '''
class A:
    """This is a class"""
    def __init__(self):
        """This is a constructor"""
        self.a = 1

    def get_a(self) -> int:
        """This is a getter"""
        return self.a

def make_aa() -> int:
    """This is a top-level function"""
    a = A()
    a = a.get_a()
    return a
'''


def _download_einops():
    """
    Download einops repository to ./temp/einops
    """
    if not os.path.exists("./temp"):
        os.makedirs("./temp")
    with open("./temp/einops_issue.md", "w") as file:
        file.write("""
print("einops") does not work
""")
    if not os.path.exists("./temp/einops"):
        os.system(
            "git clone https://github.com/arogozhnikov/einops.git --depth 1  ./temp/einops && \
            cd ./temp/einops \
            git fetch --depth 1 origin fe9d81d577dec5a224d572a5bc7d44ab8bd914eb \
            "
        )


def test_astreader():
    ar = se_gym.observe.read.ASTReader
    docs = ar._ast2doc("dummypath", dummy_file)
    for doc in docs:
        assert doc.path == "dummypath"
        assert doc.full_text == dummy_file
        assert len(doc.text) > 0
        assert len(doc.full_text) > 0
        assert len(doc.text) < len(doc.full_text)
    print(docs)


def test_observer1():
    _download_einops()
    state = se_gym.api.State(path="./temp/einops", issue="einmix is not working")
    readers = [
        se_gym.observe.read.RawReader(root_dir="./temp/einops"),
        se_gym.observe.read.OracleReader(
            root_dir="./temp/einops",
            files=[
                "./temp/einops/einops/__init__.py",
                "./temp/einops/einops/layers/_einmix.py",
            ],
        ),
        se_gym.observe.read.ASTReader(root_dir="./temp/einops"),
    ]

    selectors = [
        se_gym.observe.select.BM25Selector(),
        se_gym.observe.select.FullSelector(),
    ]

    compressors = [
        se_gym.observe.compress.NoCompression(),
    ]

    for r in readers:
        for s in selectors:
            for c in compressors:
                r.clear_cache()
                s.clear_cache()

                obs = se_gym.observe.Observer(reader=r, selector=s, compressor=c)
                result = obs(state)
                assert isinstance(result, str)
                assert len(result) > 0
