"""
main.py — Offline RAG App entry point (Kivy / Android).

Architecture
────────────
  UI     : Kivy (pure Python, Android-ready via Buildozer)
  LLM    : llama-cpp-python  (GGUF models — user supplies the model)
  Retriev: Hybrid BM25 + TF-IDF cosine (pure Python/stdlib + pickle)
  Docs   : .txt (built-in) · .pdf (PyMuPDF)
  DB     : SQLite3 (built-in)

Storage requirements
────────────────────
  App + deps : ~25 MB
  GGUF model : user-supplied  (TinyLlama Q4 ≈ 640 MB · Phi-2 Q4 ≈ 1.6 GB)
  Doc index  : <5 MB for hundreds of pages
"""

# ── Kivy config BEFORE any other kivy import ──────────────────────── #
import os
# Suppress Kivy's verbose logging
os.environ.setdefault("KIVY_LOG_LEVEL", "warning")

from kivy.config import Config
Config.set("kivy", "window_icon", "assets/icon.png")
# ─────────────────────────────────────────────────────────────────── #

from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, FadeTransition
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.metrics import dp, sp
from kivy.graphics import Color, RoundedRectangle
from kivy.clock import Clock

import sys
sys.path.insert(0, os.path.dirname(__file__))  # ensure src/ imports work

from src.ui.screens.chat_screen     import ChatScreen
from src.ui.screens.docs_screen     import DocsScreen
from src.ui.screens.settings_screen import SettingsScreen
from src.rag.pipeline               import init


# ------------------------------------------------------------------ #
#  Bottom navigation bar                                               #
# ------------------------------------------------------------------ #

_NAV_ITEMS = [
    ("chat",     "💬 Chat"),
    ("docs",     "📄 Docs"),
    ("settings", "⚙  Settings"),
]

_ACTIVE_COLOR   = (0.098, 0.761, 0.490, 1)   # ChatGPT green  #19c37d
_INACTIVE_COLOR = (0.30, 0.30, 0.33, 1)
_BG_COLOR       = (0.102, 0.102, 0.102, 1)    # #1a1a1a


class NavBar(BoxLayout):
    def __init__(self, on_switch, **kw):
        super().__init__(
            size_hint=(1, None), height=dp(56),
            orientation="horizontal",
            **kw,
        )
        with self.canvas.before:
            Color(*_BG_COLOR)
            self._bg = RoundedRectangle(radius=[0])
        self.bind(pos=lambda w, _: setattr(self._bg, "pos", w.pos),
                  size=lambda w, _: setattr(self._bg, "size", w.size))

        self._buttons: dict[str, Button] = {}
        for name, label in _NAV_ITEMS:
            btn = Button(
                text             = label,
                font_size        = sp(12),
                background_normal= "",
                background_color = _INACTIVE_COLOR,
            )
            btn.bind(on_release=lambda b, n=name: on_switch(n))
            self._buttons[name] = btn
            self.add_widget(btn)

    def set_active(self, name: str):
        for n, btn in self._buttons.items():
            btn.background_color = _ACTIVE_COLOR if n == name else _INACTIVE_COLOR


# ------------------------------------------------------------------ #
#  Root layout                                                         #
# ------------------------------------------------------------------ #

class RootLayout(BoxLayout):
    def __init__(self, **kw):
        super().__init__(orientation="vertical", **kw)

        with self.canvas.before:
            Color(0.129, 0.129, 0.129, 1)  # #212121 ChatGPT dark
            self._bg = RoundedRectangle(radius=[0])
        self.bind(pos=lambda w, _: setattr(self._bg, "pos", w.pos),
                  size=lambda w, _: setattr(self._bg, "size", w.size))

        # screen manager
        self.sm = ScreenManager(transition=FadeTransition(duration=0.15))
        self.sm.add_widget(ChatScreen(name="chat"))
        self.sm.add_widget(DocsScreen(name="docs"))
        self.sm.add_widget(SettingsScreen(name="settings"))

        self.add_widget(self.sm)  # screens fill remaining space

        # bottom nav
        self.navbar = NavBar(on_switch=self.switch_screen)
        self.add_widget(self.navbar)
        self.navbar.set_active("chat")

    def switch_screen(self, name: str):
        self.sm.current = name
        self.navbar.set_active(name)


# ------------------------------------------------------------------ #
#  App                                                                 #
# ------------------------------------------------------------------ #

class RAGApp(App):
    title = "Offline RAG"

    def build(self):
        # Initialise DB and retriever once
        Clock.schedule_once(lambda *_: init(), 0)
        return RootLayout()


if __name__ == "__main__":
    RAGApp().run()
