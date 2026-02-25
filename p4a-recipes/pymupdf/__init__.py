"""
python-for-android recipe for pymupdf.

PyMuPDF relies on a large C library (MuPDF) that is non-trivial to
cross-compile.  On Android we substitute the pure-Python 'pypdf' library
and ship a thin compatibility shim module named 'fitz' so the rest of
the codebase needs no changes.
"""
from pythonforandroid.recipe import PythonRecipe


class PyMuPDFAndroidRecipe(PythonRecipe):
    """
    Install pypdf (pure-Python) and a fitz compatibility shim
    so code that does `import fitz` still works on Android.
    """
    name    = "pymupdf"
    version = "1.24.11"

    # Don't try to download pymupdf sources — we install pypdf instead
    url  = None
    md5  = None
    sha1 = None

    # We need setuptools to install pypdf
    depends = ["python3", "setuptools", "pip"]

    call_hostpython_via_targetpython = False
    install_in_hostpython = False

    # The actual content of the shim is written in postbuild
    _FITZ_SHIM = '''"""
fitz — compatibility shim for Android.
Delegates to pypdf for text extraction.  Images / pixmaps are stubs.
"""
from __future__ import annotations
from typing import List

try:
    import pypdf as _pypdf
except ImportError:          # graceful degradation if even pypdf is absent
    _pypdf = None


class Pixmap:
    """Stub Pixmap — images are not supported in the Android shim."""
    def __init__(self, *args, **kwargs):
        self.width  = 0
        self.height = 0
        self.n      = 0
    def tobytes(self, output="png"):
        return b""
    def save(self, path):
        pass


class Page:
    def __init__(self, text: str, number: int):
        self._text   = text
        self.number  = number

    def get_text(self, kind="text") -> str:
        return self._text

    def get_pixmap(self, *args, **kwargs) -> Pixmap:
        return Pixmap()


class Document:
    def __init__(self, filename: str | None = None,
                 stream=None, filetype: str | None = None,
                 _pages: list | None = None):
        self._pages: List[Page] = _pages or []
        self._closed = False
        if filename and _pypdf:
            self._load(filename)
        elif stream and _pypdf:
            self._load_stream(stream)

    def _load(self, filename: str):
        try:
            reader = _pypdf.PdfReader(filename)
            for i, p in enumerate(reader.pages):
                self._pages.append(Page(p.extract_text() or "", i))
        except Exception as exc:
            raise RuntimeError(f"fitz shim: cannot parse PDF: {exc}") from exc

    def _load_stream(self, stream):
        import io
        try:
            reader = _pypdf.PdfReader(io.BytesIO(stream))
            for i, p in enumerate(reader.pages):
                self._pages.append(Page(p.extract_text() or "", i))
        except Exception as exc:
            raise RuntimeError(f"fitz shim: cannot parse PDF: {exc}") from exc

    def __len__(self):
        return len(self._pages)

    def __iter__(self):
        return iter(self._pages)

    def __getitem__(self, idx):
        return self._pages[idx]

    def page_count(self):
        return len(self._pages)

    def load_page(self, n: int) -> Page:
        return self._pages[n]

    def close(self):
        self._closed = True
        self._pages.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


open = Document
'''

    def build_arch(self, arch):
        from pythonforandroid.util import current_directory
        import sh
        import os

        env  = self.get_recipe_env(arch)
        hp   = self.ctx.hostpython
        site = self.ctx.get_python_install_dir(arch.arch)

        # 1. Install pypdf into the target site-packages
        sh.Command(hp)(
            "-m", "pip", "install",
            f"--target={site}",
            "--no-deps",
            "--no-build-isolation",
            "pypdf>=4.0.0",
            _env=env,
        )

        # 2. Write the fitz shim into site-packages
        fitz_dir  = os.path.join(site, "fitz")
        os.makedirs(fitz_dir, exist_ok=True)
        with open(os.path.join(fitz_dir, "__init__.py"), "w") as f:
            f.write(self._FITZ_SHIM)


recipe = PyMuPDFAndroidRecipe()
