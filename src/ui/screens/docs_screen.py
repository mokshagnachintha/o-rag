"""
docs_screen.py — Document management screen.
Tap the '+' button to open the native Android file browser and pick
a .txt or .pdf file to ingest into the RAG knowledge base.
"""
from __future__ import annotations

from kivy.uix.screenmanager import Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.clock import mainthread
from kivy.metrics import dp, sp
from kivy.uix.widget import Widget
from kivy.graphics import Color, RoundedRectangle

# ── Palette (must match chat_screen / main) ──────────────────────── #
_BG        = (0.129, 0.129, 0.129, 1)
_HEADER_BG = (0.102, 0.102, 0.102, 1)
_CARD_BG   = (0.184, 0.184, 0.184, 1)
_INPUT_BG  = (0.231, 0.231, 0.231, 1)
_GREEN     = (0.098, 0.761, 0.490, 1)
_RED       = (0.75,  0.15,  0.15,  1)
_WHITE     = (1, 1, 1, 1)
_MUTED     = (0.60, 0.60, 0.63, 1)
_DIVIDER   = (0.22, 0.22, 0.22, 1)


def _paint(widget, color, radius=0):
    with widget.canvas.before:
        Color(*color)
        rect = RoundedRectangle(radius=[dp(radius)]) if radius else RoundedRectangle()
    widget.bind(
        pos =lambda w, _: setattr(rect, "pos",  w.pos),
        size=lambda w, _: setattr(rect, "size", w.size),
    )
    return rect


class DocRow(BoxLayout):
    """One row in the documents list."""

    def __init__(self, doc: dict, on_delete, **kw):
        super().__init__(
            size_hint=(1, None), height=dp(60),
            spacing=dp(6), padding=[dp(12), dp(6)],
            **kw,
        )
        with self.canvas.before:
            Color(*_CARD_BG)
            self._bg = RoundedRectangle(radius=[dp(8)])
        self.bind(pos=lambda w,_: setattr(self._bg,'pos',w.pos),
                  size=lambda w,_: setattr(self._bg,'size',w.size))
        self.doc = doc

        info = BoxLayout(orientation="vertical", size_hint=(1, 1))
        info.add_widget(Label(
            text=doc["name"], halign="left", valign="middle",
            font_size=sp(13), color=(1, 1, 1, 1),
            text_size=(None, None), size_hint_y=None, height=dp(22),
        ))
        info.add_widget(Label(
            text=f"{doc['num_chunks']} chunks · {doc['added_at'][:16]}",
            halign="left", valign="middle",
            font_size=sp(10), color=(0.6, 0.6, 0.6, 1),
            text_size=(None, None), size_hint_y=None, height=dp(18),
        ))
        self.add_widget(info)

        del_btn = Button(
            text="✕", size_hint=(None, 1), width=dp(40),
            font_size=sp(14), background_normal="",
            background_color=(0.75, 0.15, 0.15, 1),
        )
        del_btn.bind(on_release=lambda *_: on_delete(doc["id"]))
        self.add_widget(del_btn)


class DocsScreen(Screen):
    def __init__(self, **kw):
        super().__init__(**kw)
        self._build_ui()

    def on_enter(self, *_):
        self._refresh_list()

    def _build_ui(self):
        root = BoxLayout(orientation="vertical")
        _paint(root, _BG)

        # ── header ───────────────────────────────────────────────── #
        header = BoxLayout(size_hint=(1, None), height=dp(54))
        _paint(header, _HEADER_BG)
        header.add_widget(Label(
            text="[b]Documents[/b]", markup=True,
            color=_WHITE, font_size=sp(16),
            halign="center", valign="middle",
        ))
        sep = Widget(size_hint=(1, None), height=dp(1))
        _paint(sep, _DIVIDER)
        root.add_widget(header)
        root.add_widget(sep)

        # ── doc list ─────────────────────────────────────────────── #
        self._scroll = ScrollView(size_hint=(1, 1))
        _paint(self._scroll, _BG)
        self._list = BoxLayout(
            orientation="vertical",
            size_hint=(1, None),
            spacing=dp(6),
            padding=[dp(10), dp(10)],
        )
        self._list.bind(minimum_height=self._list.setter("height"))
        self._scroll.add_widget(self._list)
        root.add_widget(self._scroll)

        # ── status label ─────────────────────────────────────────── #
        self._status = Label(
            text="", size_hint=(1, None), height=dp(28),
            font_size=sp(11), color=(0.5, 0.9, 0.5, 1),
        )
        root.add_widget(self._status)

        # ── bottom bar: file picker + manual path fallback ───────── #
        bar = BoxLayout(
            size_hint=(1, None), height=dp(72),
            orientation="vertical",
            padding=[dp(10), dp(6)],
            spacing=dp(4),
        )
        _paint(bar, _HEADER_BG)

        # Row 1 — Browse button (primary; uses native Android picker)
        browse_row = BoxLayout(size_hint=(1, None), height=dp(44), spacing=dp(8))

        browse_btn = Button(
            text             = "＋  Browse Files (PDF / TXT)",
            size_hint        = (1, 1),
            font_size        = sp(14),
            bold             = True,
            background_normal= "",
            background_color = _GREEN,
            color            = _WHITE,
        )
        _paint(browse_btn, _GREEN, radius=10)
        browse_btn.bind(on_release=self._on_browse)
        browse_row.add_widget(browse_btn)
        bar.add_widget(browse_row)

        root.add_widget(bar)

        # ── manual path row (collapsed by default, shown as fallback) #
        manual_row = BoxLayout(
            size_hint=(1, None), height=dp(48),
            spacing=dp(8), padding=[dp(10), dp(4)],
        )
        _paint(manual_row, _BG)
        self._path_input = TextInput(
            hint_text        = "Or paste a file path manually…",
            multiline        = False,
            size_hint        = (1, 1),
            font_size        = sp(12),
            foreground_color = _WHITE,
            hint_text_color  = _MUTED,
            background_color = _INPUT_BG,
            cursor_color     = _WHITE,
        )
        add_btn = Button(
            text="Add", size_hint=(None, 1), width=dp(60),
            font_size=sp(13), background_normal="",
            background_color=_GREEN,
        )
        add_btn.bind(on_release=self._on_add_manual)
        manual_row.add_widget(self._path_input)
        manual_row.add_widget(add_btn)
        root.add_widget(manual_row)

        self.add_widget(root)

    # ── helpers ──────────────────────────────────────────────────── #

    def _refresh_list(self):
        from src.rag.db import list_documents
        self._list.clear_widgets()
        docs = list_documents()
        if not docs:
            self._list.add_widget(Label(
                text="No documents yet.\nTap [b]Browse Files[/b] to add a PDF or TXT.",
                markup=True,
                size_hint=(1, None), height=dp(80),
                halign="center", font_size=sp(13),
                color=_MUTED,
            ))
        for d in docs:
            self._list.add_widget(DocRow(d, on_delete=self._on_delete))

    # ── file browser (native Android picker via plyer) ────────────── #

    def _on_browse(self, *_):
        """Open the native file chooser. Falls back to manual input if plyer unavailable."""
        try:
            from plyer import filechooser
            filechooser.open_file(
                on_selection = self._on_file_selected,
                filters      = [["Documents", "*.pdf", "*.txt", "*.PDF", "*.TXT"]],
                title        = "Choose a document",
                multiple     = False,
            )
        except Exception:
            # plyer not available — show manual path input hint
            self._set_status(
                "File browser unavailable. Use the path field below.",
                (0.9, 0.8, 0.3, 1),
            )

    def _on_file_selected(self, selection):
        """Called by plyer when the user picks a file."""
        if not selection:
            return
        path = selection[0]
        self._ingest(path)

    # ── manual path fallback ─────────────────────────────────────── #

    def _on_add_manual(self, *_):
        path = self._path_input.text.strip()
        if not path:
            return
        self._path_input.text = ""
        self._ingest(path)

    # ── shared ingest ─────────────────────────────────────────────── #

    def _ingest(self, path: str):
        import os
        name = os.path.basename(path)
        self._set_status(f"Ingesting '{name}'…", (0.9, 0.8, 0.3, 1))
        from src.rag.pipeline import ingest_document
        ingest_document(path, on_done=self._on_ingest_done)

    @mainthread
    def _on_ingest_done(self, success: bool, msg: str):
        color = (0.4, 0.9, 0.4, 1) if success else (0.9, 0.3, 0.3, 1)
        self._set_status(msg, color)
        if success:
            self._refresh_list()

    def _on_delete(self, doc_id: int):
        from src.rag.db import delete_document
        from src.rag.pipeline import retriever
        delete_document(doc_id)
        retriever.reload()
        self._refresh_list()
        self._set_status("Document removed.", (0.9, 0.6, 0.3, 1))

    def _set_status(self, text: str, color):
        self._status.text  = text
        self._status.color = color
