import abc
import warnings
import os
import glob
import typing
import ast

__all__ = ["Document", "Reader", "RawReader", "OracleReader", "ASTReader"]


class Document:
    """
    A document. Path should be a unique identifier, like a file path.
    """

    path: str
    text: str

    def __init__(
        self,
        path: str,
        text: str,
        start_line: typing.Optional[int] = None,
        end_line: typing.Optional[int] = None,
        full_text: typing.Optional[str] = None,
    ):
        self.path = path
        self.text = text
        if start_line is None:
            start_line = 0
        if end_line is None:
            end_line = len(text.split("\n"))
        if full_text is None:
            full_text = text
        self.full_text = full_text
        self.start_line = start_line
        self.end_line = end_line

    def get_formatted(self):
        with_lines = f"\n# {self.path}\n"
        for i, line in enumerate(
            self.full_text.split("\n")[self.start_line : self.end_line]
        ):
            with_lines += f"{i + self.start_line}: {line}\n"
        return with_lines


class Reader(abc.ABC):
    """
    The reader is the first element of the retrieval chain.
    It reads some or all files in some way and returns the documents.
    """

    def __init__(self):
        self.cache = []

    def clear_cache(self):
        self.cache = []

    def get_documents(self) -> typing.List[Document]:
        if self.cache:
            return self.cache
        self.cache = self._populate()
        return self.cache

    @abc.abstractmethod
    def _populate(self): ...

    @staticmethod
    @abc.abstractmethod
    def from_env(env):
        """Create a reader from the environment."""

    @staticmethod
    def _format_filepath(filepath) -> str:
        filepath = filepath.replace("\\", "/").replace("//", "/")
        if filepath.startswith("/"):
            filepath = filepath[1:]
        return filepath


class RawReader(Reader):
    """
    Reads everything in the code directory.
    """

    def __init__(self, root_dir, glob_pattern="**/*.py"):
        """
        Read all files in the directory.
        """
        super().__init__()
        self.root_dir = root_dir
        self.glob_pattern = glob_pattern
        self.path = os.path.join(root_dir, glob_pattern)

    def _populate(self):
        documents = []
        for f in glob.glob(self.path, recursive=True):
            with open(f, "r") as file:
                documents.append(Document(f.replace(self.root_dir, ""), file.read()))
        return documents

    @staticmethod
    def from_env(env):
        return RawReader(env.current_path)


class OracleReader(Reader):
    """
    Read from manually selected files.
    This reader works with select.FullSelector, as it already contains the relevant files.
    """

    def __init__(self, root_dir: str, files: typing.List[str]):
        super().__init__()
        self.files = files
        self.root_dir = root_dir

    def _populate(self):
        files_existing = []
        for file in self.files:
            if not os.path.exists(file):
                warnings.warn(f"File {file} does not exist.")
            else:
                files_existing.append(file)
        self.files = files_existing
        documents = []
        for file in files_existing:
            with open(file, "r") as openfile:
                documents.append(
                    Document(file.replace(self.root_dir, ""), openfile.read())
                )
        return documents

    @staticmethod
    def from_env(env):
        return OracleReader(
            root_dir=env.current_path,
            files=[env.current_path + "/" + f for f in env.oracle_files],
        )


class _DocstringExtractor(ast.NodeVisitor):
    """
    Helper class to extract docstrings from AST nodes.
    """

    def __init__(self):
        self.results = []

    def _get_fun(self, node):
        return {
            "type": "function",
            "name": node.name,
            "docstring": ast.get_docstring(node),
            "args": [ast.unparse(arg) for arg in node.args.args],
            "returns": ast.unparse(node.returns) if node.returns else None,
            "lineno": node.lineno - 1,
            "end_lineno": node.end_lineno,
        }

    def visit_ClassDef(self, node):
        class_info = {
            "type": "class",
            "name": node.name,
            "docstring": ast.get_docstring(node),
            "methods": [],
            "lineno": node.lineno - 1,
            "end_lineno": node.end_lineno,
        }
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                class_info["methods"].append(self._get_fun(item))
        self.results.append(class_info)
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        if isinstance(node.parent, ast.Module):
            self.results.append(self._get_fun(node))
        self.generic_visit(node)


class ASTReader(Reader):
    """
    Use AST to parse the code.
    Documents are containing the identifiers as paths and the method signatures and docs as text.
    """

    INDENT = "    "

    def __init__(self, root_dir, glob_pattern="**/*.py"):
        """
        Read all files in the directory.
        """
        super().__init__()
        self.root_dir = root_dir
        self.glob_pattern = glob_pattern
        self.path = os.path.join(root_dir, glob_pattern)

    @staticmethod
    def add_parents(node):
        """Make sure each node has a reference to its parent."""
        for child in ast.iter_child_nodes(node):
            child.parent = node
            ASTReader.add_parents(child)

    @staticmethod
    def _ast2doc(filename, filefull):
        tree = ast.parse(filefull)
        ASTReader.add_parents(tree)
        extractor = _DocstringExtractor()
        extractor.visit(tree)
        docs = []
        for item in extractor.results:
            docargs = dict(
                path=filename,
                start_line=item["lineno"],
                end_line=item["end_lineno"],
                full_text=filefull,
            )
            if item["type"] == "class":
                docs.append(Document(text=ASTReader._class2doctxt(item), **docargs))
            elif item["type"] == "function":
                docs.append(Document(text=ASTReader._function2doctxt(item), **docargs))
        return docs

    @staticmethod
    def _class2doctxt(item):
        assert item["type"] == "class", f"Item is not a class but {item['type']}."
        doctxts = [f"class {item['name']}:"]
        if item["docstring"]:
            doctxts.append(f'{ASTReader.INDENT}"""{item["docstring"]}"""')
        for method in item["methods"]:
            doctxts.append(ASTReader._function2doctxt(method, indent=1))
        return "\n".join(doctxts)

    @staticmethod
    def _function2doctxt(item, indent=0):
        assert item["type"] == "function", f"Item is not a function but {item['type']}."
        doctxts = []
        args = ", ".join(item["args"])
        if item["returns"]:
            doctxts.append(
                f"{ASTReader.INDENT * indent}def {item['name']}({args}) -> {item['returns']}:"
            )
        else:
            doctxts.append(f"{ASTReader.INDENT * indent}def {item['name']}({args}):")
        if item["docstring"]:
            doctxts.append(f'{ASTReader.INDENT * (indent+1)}"""{item["docstring"]}"""')
        return "\n".join(doctxts)

    def _populate(self):
        documents = []
        for f in glob.glob(self.path, recursive=True):
            with open(f, "r") as file:
                filefull = file.read()
            documents.extend(
                ASTReader._ast2doc(
                    self._format_filepath(f.replace(self.root_dir, "")), filefull
                )
            )
        return documents

    @staticmethod
    def from_env(env):
        return ASTReader(env.current_path)


class Summarizer(Reader):
    """
    Let an LLM create a code map, summarizing the contents of each file, then each directory.
    This reader is trainable.
    """


class CompressedReader(Reader):
    """
    Use an LLM to create an semantic compression of the code.
    """
