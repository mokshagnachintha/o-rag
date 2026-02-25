"""
docs_screen.py — Document management screen.
Users can see ingested documents and add new ones (via file path input).
On Android, documents should be placed in:
    /storage/emulated/0/Documents/  (or use a file chooser popup)
"""
from __future__ import annotations

from kivy.uix.screenmanager import Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.popup import Popup
from kivy.clock import mainthread
from kivy.metrics import dp, sp
from kivy.uix.widget import Widget
from kivy.graphics import Color, RoundedRectangle


class DocRow(BoxLayout):
    """One row in the documents list."""

    def __init__(self, doc: dict, on_delete, **kw):
        super().__init__(
            size_hint=(1, None), height=dp(60),
            spacing=dp(6), padding=[dp(12), dp(6)],
            **kw,
        )
        with self.canvas.before:
            Color(0.184, 0.184, 0.184, 1)
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

        # header
        header = BoxLayout(size_hint=(1, None), height=dp(54))
        with header.canvas.before:
            Color(0.102, 0.102, 0.102, 1)
            self._hdr_rect = RoundedRectangle(radius=[0])
        header.bind(pos=lambda w, _: setattr(self._hdr_rect, "pos", w.pos),
                    size=lambda w, _: setattr(self._hdr_rect, "size", w.size))
        header.add_widget(Label(
            text="[b]Documents[/b]", markup=True,
            color=(1, 1, 1, 1), font_size=sp(16),
            halign="center", valign="middle",
        ))
        sep = Widget(size_hint=(1, None), height=dp(1))
        with sep.canvas.before:
            Color(0.22, 0.22, 0.22, 1)
            sep._r = RoundedRectangle()
        sep.bind(pos=lambda w,_: setattr(w._r,'pos',w.pos), size=lambda w,_: setattr(w._r,'size',w.size))
        root.add_widget(header)
        root.add_widget(sep)

        # page background
        with root.canvas.before:
            Color(0.129, 0.129, 0.129, 1)
            self._root_bg = RoundedRectangle()
        root.bind(pos=lambda w,_: setattr(self._root_bg,'pos',w.pos),
                  size=lambda w,_: setattr(self._root_bg,'size',w.size))

        # doc list
        self._scroll = ScrollView(size_hint=(1, 1))
        with self._scroll.canvas.before:
            Color(0.129, 0.129, 0.129, 1)
            self._scroll_bg = RoundedRectangle()
        self._scroll.bind(pos=lambda w,_: setattr(self._scroll_bg,'pos',w.pos),
                          size=lambda w,_: setattr(self._scroll_bg,'size',w.size))
        self._list = BoxLayout(
            orientation="vertical",
            size_hint=(1, None),
            spacing=dp(4),
            padding=[dp(8), dp(8)],
        )
        self._list.bind(minimum_height=self._list.setter("height"))
        self._scroll.add_widget(self._list)
        root.add_widget(self._scroll)

        # status label
        self._status = Label(
            text="", size_hint=(1, None), height=dp(28),
            font_size=sp(11), color=(0.5, 0.9, 0.5, 1),
        )
        root.add_widget(self._status)

        # add-doc bar
        bar = BoxLayout(
            size_hint=(1, None), height=dp(64),
            spacing=dp(8), padding=[dp(12), dp(8)],
        )
        with bar.canvas.before:
            Color(0.102, 0.102, 0.102, 1)
            self._bar_rect = RoundedRectangle(radius=[0])
        bar.bind(pos=lambda w, _: setattr(self._bar_rect, "pos", w.pos),
                 size=lambda w, _: setattr(self._bar_rect, "size", w.size))

        self._path_input = TextInput(
            hint_text="Full path to .txt or .pdf file",
            multiline=False,
            size_hint=(1, 1),
            font_size=sp(13),
            foreground_color=(1, 1, 1, 1),
            hint_text_color=(0.60, 0.60, 0.63, 1),
            background_color=(0.231, 0.231, 0.231, 1),
            cursor_color=(1, 1, 1, 1),
        )
        add_btn = Button(
            text="Add", size_hint=(None, 1), width=dp(64),
            font_size=sp(13), background_normal="",
            background_color=(0.098, 0.761, 0.490, 1),
        )
        add_btn.bind(on_release=self._on_add)
        bar.add_widget(self._path_input)
        bar.add_widget(add_btn)
        root.add_widget(bar)

        self.add_widget(root)

    # ---- helpers ----

    def _refresh_list(self):
        from src.rag.db import list_documents
        self._list.clear_widgets()
        docs = list_documents()
        if not docs:
            self._list.add_widget(Label(
                text="No documents yet.\nAdd a .txt or .pdf file below.",
                size_hint=(1, None), height=dp(80),
                halign="center", font_size=sp(13),
                color=(0.6, 0.6, 0.6, 1),
            ))
        for d in docs:
            self._list.add_widget(DocRow(d, on_delete=self._on_delete))

    def _on_add(self, *_):
        path = self._path_input.text.strip()
        if not path:
            return
        self._set_status("Ingesting…", (0.9, 0.8, 0.3, 1))
        from src.rag.pipeline import ingest_document
        ingest_document(path, on_done=self._on_ingest_done)

    @mainthread
    def _on_ingest_done(self, success: bool, msg: str):
        color = (0.4, 0.9, 0.4, 1) if success else (0.9, 0.3, 0.3, 1)
        self._set_status(msg, color)
        if success:
            self._path_input.text = ""
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
