"""
main.py — Offline RAG App entry point (Kivy / Android).

Single-screen design: one chat interface.
  • Tap + to attach a PDF or TXT document (RAG mode activates automatically)
  • Otherwise chat freely with the AI
  • Model is bundled in the APK — extracted to device storage on first launch
"""

# ── Kivy config BEFORE any other kivy import ──────────────────────── #
import os
os.environ.setdefault("KIVY_LOG_LEVEL", "warning")

from kivy.config import Config
Config.set("kivy", "window_icon", "assets/icon.png")
# ─────────────────────────────────────────────────────────────────── #

# Keep input bar visible above the soft keyboard on Android
from kivy.core.window import Window
Window.softinput_mode = "below_target"

from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, FadeTransition
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics import Color, Rectangle
from kivy.clock import Clock

import sys
sys.path.insert(0, os.path.dirname(__file__))

from src.ui.screens.chat_screen import ChatScreen
from src.rag.pipeline           import init


class RAGApp(App):
    title = "Offline RAG"

    def build(self):
        root = BoxLayout(orientation="vertical")
        with root.canvas.before:
            Color(0.102, 0.102, 0.102, 1)   # #1a1a1a
            bg = Rectangle()
        root.bind(
            pos =lambda w, _: setattr(bg, "pos",  w.pos),
            size=lambda w, _: setattr(bg, "size", w.size),
        )

        sm = ScreenManager(transition=FadeTransition(duration=0.12))
        sm.add_widget(ChatScreen(name="chat"))
        root.add_widget(sm)

        # Init DB + retriever, then kick off model loading (bundled or download)
        Clock.schedule_once(lambda *_: init(), 0)
        return root


if __name__ == "__main__":
    RAGApp().run()

