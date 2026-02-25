"""
chat_screen.py — Chat / Q-A interface with streaming token output.
"""
from __future__ import annotations

from kivy.uix.screenmanager import Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.clock import Clock, mainthread
from kivy.metrics import dp, sp
from kivy.graphics import Color, RoundedRectangle


# ------------------------------------------------------------------ #
#  Bubble widget                                                       #
# ------------------------------------------------------------------ #

class MessageBubble(BoxLayout):
    """A single chat bubble (user or assistant)."""

    COLORS = {
        "user":      (0.27, 0.51, 0.96, 1),   # blue
        "assistant": (0.18, 0.18, 0.22, 1),   # dark grey
        "system":    (0.90, 0.45, 0.10, 1),   # orange (warnings)
    }

    def __init__(self, text: str, role: str = "user", **kw):
        super().__init__(orientation="vertical", size_hint=(1, None), **kw)
        self.role    = role
        self._label  = Label(
            text             = text,
            markup           = True,
            size_hint_y      = None,
            text_size        = (None, None),
            halign           = "left",
            valign           = "top",
            color            = (1, 1, 1, 1),
            font_size        = sp(14),
            padding          = (dp(12), dp(8)),
        )
        self._label.bind(texture_size=self._on_texture)
        bubble = BoxLayout(size_hint=(1, None))
        bubble.add_widget(self._label)
        self.add_widget(bubble)
        self._draw_bg(bubble)
        self.bind(width=self._on_width)

    def _draw_bg(self, container):
        color = self.COLORS.get(self.role, self.COLORS["system"])
        with container.canvas.before:
            Color(*color)
            self._rect = RoundedRectangle(radius=[dp(10)])
        container.bind(pos=self._upd_rect, size=self._upd_rect)

    def _upd_rect(self, obj, *_):
        self._rect.pos  = obj.pos
        self._rect.size = obj.size

    def _on_texture(self, lbl, tex_size):
        lbl.height = tex_size[1] + dp(16)
        lbl.text_size = (lbl.width, None)
        self.height = lbl.height + dp(6)

    def _on_width(self, *_):
        self._label.text_size = (self.width - dp(24), None)

    def append(self, token: str):
        self._label.text += token


# ------------------------------------------------------------------ #
#  Chat Screen                                                         #
# ------------------------------------------------------------------ #

class ChatScreen(Screen):
    def __init__(self, **kw):
        super().__init__(**kw)
        self._current_bubble: MessageBubble | None = None
        self._build_ui()

    # ---- layout ----

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
            text="[b]RAG Chat[/b]", markup=True,
            color=(1, 1, 1, 1), font_size=sp(16)
        ))
        root.add_widget(header)

        # ---- message list ----
        self._scroll = ScrollView(size_hint=(1, 1))
        self._msg_list = BoxLayout(
            orientation="vertical",
            size_hint=(1, None),
            spacing=dp(6),
            padding=[dp(8), dp(8)],
        )
        self._msg_list.bind(minimum_height=self._msg_list.setter("height"))
        self._scroll.add_widget(self._msg_list)
        root.add_widget(self._scroll)

        # ---- input bar ----
        bar = BoxLayout(
            size_hint=(1, None), height=dp(56),
            spacing=dp(6), padding=[dp(8), dp(6)],
        )
        with bar.canvas.before:
            Color(0.13, 0.13, 0.16, 1)
            self._bar_rect = RoundedRectangle(radius=[0])
        bar.bind(pos=lambda w, _: setattr(self._bar_rect, "pos", w.pos),
                 size=lambda w, _: setattr(self._bar_rect, "size", w.size))

        self._input = TextInput(
            hint_text   = "Ask a question about your documents…",
            multiline   = False,
            size_hint   = (1, 1),
            font_size   = sp(13),
            foreground_color = (1, 1, 1, 1),
            background_color = (0.20, 0.20, 0.25, 1),
            cursor_color     = (1, 1, 1, 1),
        )
        self._input.bind(on_text_validate=self._on_send)

        send_btn = Button(
            text="Send",
            size_hint=(None, 1),
            width=dp(70),
            font_size=sp(13),
            background_color=(0.27, 0.51, 0.96, 1),
            background_normal="",
        )
        send_btn.bind(on_release=self._on_send)
        bar.add_widget(self._input)
        bar.add_widget(send_btn)
        root.add_widget(bar)

        self.add_widget(root)

    # ---- helpers ----

    def _add_bubble(self, text: str, role: str) -> MessageBubble:
        b = MessageBubble(text, role=role)
        self._msg_list.add_widget(b)
        Clock.schedule_once(lambda *_: self._scroll_down(), 0.05)
        return b

    def _scroll_down(self):
        self._scroll.scroll_y = 0

    # ---- events ----

    def _on_send(self, *_):
        question = self._input.text.strip()
        if not question:
            return
        self._input.text = ""
        self._add_bubble(question, role="user")

        # placeholder bubble for streaming
        self._current_bubble = self._add_bubble("", role="assistant")
        self._current_bubble._label.text = "…"

        # kick off async
        from src.rag.pipeline import ask
        ask(
            question,
            top_k=4,
            stream_cb=self._on_token,
            on_done=self._on_done,
        )

    @mainthread
    def _on_token(self, token: str):
        if self._current_bubble:
            if self._current_bubble._label.text == "…":
                self._current_bubble._label.text = ""
            self._current_bubble.append(token)
            Clock.schedule_once(lambda *_: self._scroll_down(), 0.02)

    @mainthread
    def _on_done(self, success: bool, message: str):
        if not success:
            if self._current_bubble:
                self._current_bubble._label.text = f"[color=ff4444]{message}[/color]"
            else:
                self._add_bubble(message, role="system")
        self._current_bubble = None
