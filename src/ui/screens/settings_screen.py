"""
settings_screen.py — Model management:
   Browse & download Gemma GGUF models straight from Hugging Face
   See download progress live
   Load / unload models into llama-cpp-python
"""
from __future__ import annotations

import os
from pathlib import Path

from kivy.uix.screenmanager import Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.progressbar import ProgressBar
from kivy.uix.textinput import TextInput
from kivy.clock import mainthread
from kivy.metrics import dp, sp
from kivy.graphics import Color, RoundedRectangle

from src.rag.downloader import GEMMA_MODELS, is_downloaded, download_model


# ------------------------------------------------------------------ #
#  Model row widget                                                    #
# ------------------------------------------------------------------ #

class ModelRow(BoxLayout):
    """
    One row in the Gemma catalogue.

    States:
         "ready"       — file already in models/  show Load button
         "download"    — not yet downloaded  show Download button
         "downloading" — in-progress  show progress bar
         "loaded"      — currently loaded in llama.cpp
    """

    def __init__(self, meta: dict, on_load, **kw):
        super().__init__(
            orientation="vertical",
            size_hint=(1, None),
            padding=[dp(10), dp(6)],
            spacing=dp(4),
            **kw,
        )
        self.meta    = meta
        self.on_load = on_load
        self._draw_bg()
        self._build()
        self.height = dp(72)

    def _draw_bg(self):
        with self.canvas.before:
            Color(0.15, 0.15, 0.19, 1)
            self._bg = RoundedRectangle(radius=[dp(8)])
        self.bind(pos=lambda w, _: setattr(self._bg, "pos", w.pos),
                  size=lambda w, _: setattr(self._bg, "size", w.size))

    def _build(self):
        # top row: name + size tag
        top = BoxLayout(size_hint=(1, None), height=dp(24))
        top.add_widget(Label(
            text=self.meta["label"],
            halign="left", valign="middle",
            font_size=sp(12), color=(1, 1, 1, 1),
            text_size=(None, None), size_hint=(1, 1),
        ))
        self.add_widget(top)

        # bottom row: status label + action button
        bottom = BoxLayout(size_hint=(1, None), height=dp(34), spacing=dp(6))

        self._status_lbl = Label(
            text="", size_hint=(1, 1),
            font_size=sp(10), halign="left", valign="middle",
            color=(0.65, 0.65, 0.65, 1),
        )
        bottom.add_widget(self._status_lbl)

        self._prog = ProgressBar(
            max=100, value=0,
            size_hint=(1, None), height=dp(8),
        )

        self._btn = Button(
            size_hint=(None, 1), width=dp(100),
            font_size=sp(12), background_normal="",
        )
        self._btn.bind(on_release=self._on_btn)
        bottom.add_widget(self._btn)
        self.add_widget(bottom)

        self._bottom = bottom
        self.refresh_state()

    def refresh_state(self):
        fname = self.meta["filename"]
        from src.rag.llm import llm

        if is_downloaded(fname):
            loaded = llm.is_loaded() and Path(llm._model_path or "").name == fname
            if loaded:
                self._status_lbl.text  = "Loaded "
                self._status_lbl.color = (0.4, 0.9, 0.4, 1)
                self._btn.text             = "Loaded"
                self._btn.background_color = (0.25, 0.55, 0.25, 1)
                self._btn.disabled         = True
            else:
                self._status_lbl.text  = "Downloaded"
                self._status_lbl.color = (0.55, 0.55, 0.55, 1)
                self._btn.text             = "Load"
                self._btn.background_color = (0.27, 0.51, 0.96, 1)
                self._btn.disabled         = False
        else:
            self._status_lbl.text  = f"~{self.meta['size_mb']} MB  not downloaded"
            self._status_lbl.color = (0.55, 0.55, 0.55, 1)
            self._btn.text             = "Download"
            self._btn.background_color = (0.20, 0.55, 0.35, 1)
            self._btn.disabled         = False

    # ---- progress bar helpers ----

    def _show_progress(self):
        if self._prog not in self._bottom.children:
            self._bottom.remove_widget(self._btn)
            self._bottom.add_widget(self._prog)

    def _hide_progress(self):
        if self._prog in self._bottom.children:
            self._bottom.remove_widget(self._prog)
            self._bottom.add_widget(self._btn)

    # ---- button handler ----

    def _on_btn(self, *_):
        fname = self.meta["filename"]
        if is_downloaded(fname):
            from src.rag.downloader import model_dest_path
            self.on_load(model_dest_path(fname), self._on_load_done)
            self._btn.text     = "Loading..."
            self._btn.disabled = True
        else:
            self._btn.disabled = True
            self._btn.text     = "..."
            self._show_progress()
            self._status_lbl.text  = "Connecting..."
            self._status_lbl.color = (0.9, 0.8, 0.3, 1)
            download_model(
                repo_id     = self.meta["repo_id"],
                filename    = self.meta["filename"],
                on_progress = self._on_progress,
                on_done     = self._on_download_done,
            )

    @mainthread
    def _on_progress(self, fraction: float, text: str):
        self._prog.value      = int(fraction * 100)
        self._status_lbl.text = text

    @mainthread
    def _on_download_done(self, success: bool, msg: str):
        self._hide_progress()
        if success:
            self._status_lbl.text  = "Downloaded"
            self._status_lbl.color = (0.4, 0.9, 0.4, 1)
        else:
            self._status_lbl.text  = msg
            self._status_lbl.color = (0.9, 0.3, 0.3, 1)
        self.refresh_state()

    @mainthread
    def _on_load_done(self, success: bool, msg: str):
        self.refresh_state()


# ------------------------------------------------------------------ #
#  Settings Screen                                                     #
# ------------------------------------------------------------------ #

class SettingsScreen(Screen):
    def __init__(self, **kw):
        super().__init__(**kw)
        self._rows: list = []
        self._build_ui()
        # Hook into the pipeline auto-download that started at app launch
        from src.rag.pipeline import register_auto_download_callbacks
        register_auto_download_callbacks(
            on_progress=self._on_auto_progress,
            on_done=self._on_auto_done,
        )

    def on_enter(self, *_):
        self._update_model_status()
        for row in self._rows:
            row.refresh_state()

    def _build_ui(self):
        root = BoxLayout(orientation="vertical")

        # ---- header ----
        header = BoxLayout(size_hint=(1, None), height=dp(50))
        with header.canvas.before:
            Color(0.13, 0.13, 0.16, 1)
            self._hdr_rect = RoundedRectangle(radius=[0])
        header.bind(pos=lambda w, _: setattr(self._hdr_rect, "pos", w.pos),
                    size=lambda w, _: setattr(self._hdr_rect, "size", w.size))
        header.add_widget(Label(
            text="[b]Models[/b]", markup=True,
            color=(1, 1, 1, 1), font_size=sp(16),
        ))
        root.add_widget(header)

        # ---- auto-download banner (collapsed when idle) ----
        self._dl_banner = BoxLayout(
            orientation="vertical",
            size_hint=(1, None), height=0,
            padding=[dp(10), dp(4)],
            spacing=dp(2),
        )
        with self._dl_banner.canvas.before:
            Color(0.08, 0.22, 0.38, 1)
            self._dl_bg = RoundedRectangle(radius=[0])
        self._dl_banner.bind(
            pos=lambda w, _: setattr(self._dl_bg, "pos", w.pos),
            size=lambda w, _: setattr(self._dl_bg, "size", w.size),
        )
        self._dl_lbl = Label(
            text="Downloading Gemma 3 4B Q4...",
            size_hint=(1, None), height=dp(20),
            font_size=sp(11), halign="left", valign="middle",
            color=(0.85, 0.92, 1.0, 1),
        )
        self._dl_prog = ProgressBar(
            max=100, value=0,
            size_hint=(1, None), height=dp(8),
        )
        self._dl_banner.add_widget(self._dl_lbl)
        self._dl_banner.add_widget(self._dl_prog)
        root.add_widget(self._dl_banner)

        # ---- current model status + unload ----
        status_row = BoxLayout(size_hint=(1, None), height=dp(46), spacing=dp(8),
                               padding=[dp(10), dp(4)])
        self._model_lbl = Label(
            text="No model loaded.",
            size_hint=(1, 1), font_size=sp(12),
            halign="left", valign="middle",
            color=(0.9, 0.6, 0.3, 1),
        )
        status_row.add_widget(self._model_lbl)
        unload_btn = Button(
            text="Unload", size_hint=(None, 1), width=dp(72),
            font_size=sp(12), background_normal="",
            background_color=(0.55, 0.15, 0.15, 1),
        )
        unload_btn.bind(on_release=self._on_unload)
        status_row.add_widget(unload_btn)
        root.add_widget(status_row)

        # ---- section label ----
        root.add_widget(Label(
            text="  Gemma models (tap Download then Load):",
            size_hint=(1, None), height=dp(26),
            font_size=sp(11), halign="left",
            color=(0.6, 0.6, 0.6, 1),
        ))

        # ---- Gemma model catalogue ----
        scroll = ScrollView(size_hint=(1, 1))
        self._list = BoxLayout(
            orientation="vertical",
            size_hint=(1, None),
            spacing=dp(6),
            padding=[dp(8), dp(6)],
        )
        self._list.bind(minimum_height=self._list.setter("height"))

        for meta in GEMMA_MODELS:
            row = ModelRow(meta, on_load=self._load_model)
            self._rows.append(row)
            self._list.add_widget(row)

        scroll.add_widget(self._list)
        root.add_widget(scroll)

        # ---- manual path ----
        root.add_widget(Label(
            text="  Or load a local .gguf file manually:",
            size_hint=(1, None), height=dp(24),
            font_size=sp(11), halign="left",
            color=(0.6, 0.6, 0.6, 1),
        ))
        bar = BoxLayout(
            size_hint=(1, None), height=dp(52),
            spacing=dp(6), padding=[dp(8), dp(6)],
        )
        with bar.canvas.before:
            Color(0.13, 0.13, 0.16, 1)
            self._bar_bg = RoundedRectangle(radius=[0])
        bar.bind(pos=lambda w, _: setattr(self._bar_bg, "pos", w.pos),
                 size=lambda w, _: setattr(self._bar_bg, "size", w.size))
        self._path_in = TextInput(
            hint_text="Full path to .gguf file...",
            multiline=False, size_hint=(1, 1), font_size=sp(12),
            foreground_color=(1, 1, 1, 1),
            background_color=(0.20, 0.20, 0.25, 1),
        )
        load_btn = Button(
            text="Load", size_hint=(None, 1), width=dp(58),
            font_size=sp(12), background_normal="",
            background_color=(0.27, 0.51, 0.96, 1),
        )
        load_btn.bind(on_release=lambda *_: self._load_model(
            self._path_in.text.strip(), self._on_manual_load_done
        ))
        bar.add_widget(self._path_in)
        bar.add_widget(load_btn)
        root.add_widget(bar)

        self._status = Label(
            text="", size_hint=(1, None), height=dp(26),
            font_size=sp(11), color=(0.5, 0.9, 0.5, 1),
        )
        root.add_widget(self._status)
        self.add_widget(root)

    # ---- helpers ----

    def _update_model_status(self):
        from src.rag.llm import llm
        if llm.is_loaded():
            name = Path(llm._model_path or "").name
            self._model_lbl.text  = f"Loaded: {name}"
            self._model_lbl.color = (0.4, 0.9, 0.4, 1)
        else:
            self._model_lbl.text  = "No model loaded."
            self._model_lbl.color = (0.9, 0.6, 0.3, 1)

    def _load_model(self, path: str, callback=None):
        if not path:
            return
        self._set_status("Loading model... (may take a moment)", (0.9, 0.8, 0.3, 1))
        from src.rag.pipeline import load_model
        load_model(path, on_done=callback or self._on_manual_load_done)

    @mainthread
    def _on_manual_load_done(self, success: bool, msg: str):
        color = (0.4, 0.9, 0.4, 1) if success else (0.9, 0.3, 0.3, 1)
        self._set_status(msg, color)
        self._update_model_status()
        for row in self._rows:
            row.refresh_state()

    def _on_unload(self, *_):
        from src.rag.llm import llm
        llm.unload()
        self._update_model_status()
        self._set_status("Model unloaded.", (0.9, 0.6, 0.3, 1))
        for row in self._rows:
            row.refresh_state()

    def _set_status(self, text: str, color):
        self._status.text  = text
        self._status.color = color

    # ---- auto-download callbacks (called from background thread) ----

    @mainthread
    def _on_auto_progress(self, fraction: float, text: str):
        """Show/update the collapsible download banner."""
        if self._dl_banner.height == 0:
            self._dl_banner.height = dp(52)
        self._dl_prog.value   = int(fraction * 100)
        self._dl_lbl.text     = f"Auto-download: {text}"

    @mainthread
    def _on_auto_done(self, success: bool, message: str):
        """Collapse the banner and refresh all rows."""
        self._dl_banner.height = 0
        color = (0.4, 0.9, 0.4, 1) if success else (0.9, 0.3, 0.3, 1)
        self._set_status(message, color)
        self._update_model_status()
        for row in self._rows:
            row.refresh_state()
