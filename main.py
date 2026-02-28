"""
main.py — Kivy entry point for O-RAG Android app.

Bootstraps the Kivy App and hands off to the ChatScreen (ui/screens/chat_screen.py),
which contains the full UI and the model-loading / RAG pipeline.
"""
import os
import sys

# Ensure the app root is on the Python path so that `rag.*` and `ui.*` imports work
# both on Android (ANDROID_PRIVATE) and on desktop.
_APP_ROOT = os.path.dirname(os.path.abspath(__file__))
if _APP_ROOT not in sys.path:
    sys.path.insert(0, _APP_ROOT)

# Kivy window setup — must happen before any other kivy import
from kivy.config import Config
Config.set("graphics", "resizable", "0")

from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, NoTransition

from ui.screens.chat_screen import ChatScreen


class OragApp(App):
    """Root Kivy application."""

    def build(self):
        self.title = "O-RAG"
        sm = ScreenManager(transition=NoTransition())
        sm.add_widget(ChatScreen(name="chat"))
        return sm


if __name__ == "__main__":
    OragApp().run()

