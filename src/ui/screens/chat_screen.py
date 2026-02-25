"""
chat_screen.py — ChatGPT-style chat interface with streaming token output.
"""
from __future__ import annotations

from kivy.uix.screenmanager import Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.widget import Widget
from kivy.clock import Clock, mainthread
from kivy.metrics import dp, sp
from kivy.graphics import Color, RoundedRectangle, Rectangle

# ── Palette (ChatGPT dark theme) ─────────────────────────────────── #
_BG          = (0.129, 0.129, 0.129, 1)   # #212121  page background
_HEADER_BG   = (0.102, 0.102, 0.102, 1)   # #1a1a1a  header
_USER_BG     = (0.184, 0.184, 0.184, 1)   # #2f2f2f  user bubble
_INPUT_BG    = (0.231, 0.231, 0.231, 1)   # #3b3b3b  input field
_SEND_BG     = (0.098, 0.761, 0.490, 1)   # #19c37d  send button (ChatGPT green)
_TEXT_WHITE  = (1, 1, 1, 1)
_TEXT_MUTED  = (0.60, 0.60, 0.63, 1)
_DIVIDER     = (0.22, 0.22, 0.22, 1)


def _paint_bg(widget, color, radius=0):
    """Draw a solid color background on canvas.before."""
    with widget.canvas.before:
        Color(*color)
        rect = RoundedRectangle(radius=[dp(radius)]) if radius else Rectangle()
    def _upd(*_):
        rect.pos  = widget.pos
        rect.size = widget.size
    widget.bind(pos=_upd, size=_upd)
    return rect


# ------------------------------------------------------------------ #
#  Avatar circle (letters "U" / "AI")                                 #
# ------------------------------------------------------------------ #

class _Avatar(Widget):
    _COLORS = {
        "user":      (0.400, 0.400, 0.900, 1),
        "assistant": (0.098, 0.761, 0.490, 1),
        "system":    (0.800, 0.200, 0.200, 1),
    }
    def __init__(self, role: str, **kw):
        super().__init__(size_hint=(None, None),
                         size=(dp(32), dp(32)), **kw)
        letter = {"user": "U", "assistant": "AI", "system": "!"}.get(role, "?")
        with self.canvas:
            Color(*self._COLORS.get(role, (0.5, 0.5, 0.5, 1)))
            self._circ = RoundedRectangle(radius=[dp(16)])
        self.bind(pos=self._upd, size=self._upd)
        self._lbl = Label(text=letter, font_size=sp(11), bold=True,
                          color=(1, 1, 1, 1))
        self.add_widget(self._lbl)

    def _upd(self, *_):
        self._circ.pos  = self.pos
        self._circ.size = self.size
        self._lbl.center = self.center


# ------------------------------------------------------------------ #
#  Message row                                                         #
# ------------------------------------------------------------------ #

class MessageRow(BoxLayout):
    """
    One full-width row containing avatar + text, matching ChatGPT layout:
      - user:      right-aligned bubble, grey background
      - assistant: left-aligned, no bubble, just text on page bg
    """
    def __init__(self, text: str, role: str = "assistant", **kw):
        super().__init__(
            orientation="horizontal",
            size_hint=(1, None),
            padding=[dp(12), dp(8), dp(12), dp(8)],
            spacing=dp(10),
            **kw,
        )
        self.role = role

        self._label = Label(
            text        = text,
            markup      = True,
            size_hint_y = None,
            text_size   = (None, None),
            halign      = "left",
            valign      = "top",
            color       = _TEXT_WHITE,
            font_size   = sp(14.5),
        )
        self._label.bind(texture_size=self._on_texture)
        self.bind(width=self._on_width)

        if role == "user":
            self._build_user()
        else:
            self._build_assistant()

        _paint_bg(self, _BG if role == "assistant" else _BG)

    # -- user row: spacer | bubble(text) | avatar --
    def _build_user(self):
        self.add_widget(Widget(size_hint_x=1))          # push right
        bubble = BoxLayout(
            orientation="vertical",
            size_hint=(None, None),
            padding=[dp(12), dp(10)],
        )
        _paint_bg(bubble, _USER_BG, radius=18)
        bubble.add_widget(self._label)
        self._bubble = bubble
        self.add_widget(bubble)
        self.add_widget(_Avatar("user"))

    # -- assistant row: avatar | text --
    def _build_assistant(self):
        self.add_widget(_Avatar("assistant"))
        self.add_widget(self._label)

    def _on_texture(self, lbl, tex_size):
        lbl.height     = tex_size[1] + dp(4)
        lbl.text_size  = (lbl.width or 1, None)
        # bubble must also grow
        if self.role == "user" and hasattr(self, "_bubble"):
            self._bubble.width  = min(lbl.texture_size[0] + dp(28),
                                      self.width * 0.80)
            self._bubble.height = lbl.height + dp(20)
        self.height = max(lbl.height + dp(20), dp(52))

    def _on_width(self, *_):
        avail = self.width - dp(72)          # minus avatar + padding
        if self.role == "user":
            max_w = avail * 0.82
            self._label.text_size = (max_w, None)
        else:
            self._label.text_size = (avail, None)

    def append(self, token: str):
        self._label.text += token


# ------------------------------------------------------------------ #
#  Typing indicator ("● ● ●" animated)                                #
# ------------------------------------------------------------------ #

class _TypingIndicator(BoxLayout):
    def __init__(self, **kw):
        super().__init__(
            orientation="horizontal",
            size_hint=(1, None), height=dp(40),
            padding=[dp(56), dp(4)], spacing=dp(6),
            **kw,
        )
        self._dots = []
        for _ in range(3):
            lbl = Label(text="●", font_size=sp(10), color=_TEXT_MUTED,
                        size_hint=(None, None), size=(dp(14), dp(14)))
            self._dots.append(lbl)
            self.add_widget(lbl)
        self._tick = 0
        Clock.schedule_interval(self._animate, 0.45)

    def _animate(self, *_):
        for i, d in enumerate(self._dots):
            d.color = _TEXT_WHITE if i == self._tick % 3 else _TEXT_MUTED
        self._tick += 1

    def stop(self):
        Clock.unschedule(self._animate)


# ------------------------------------------------------------------ #
#  Chat Screen                                                         #
# ------------------------------------------------------------------ #

class ChatScreen(Screen):
    def __init__(self, **kw):
        super().__init__(**kw)
        self._current_row: MessageRow | None = None
        self._typing_indicator: _TypingIndicator | None = None
        self._build_ui()

    # ── layout ────────────────────────────────────────────────────── #

    def _build_ui(self):
        root = BoxLayout(orientation="vertical")
        _paint_bg(root, _BG)

        # ── Header ────────────────────────────────────────────────── #
        header = BoxLayout(size_hint=(1, None), height=dp(54),
                           padding=[dp(16), 0])
        _paint_bg(header, _HEADER_BG)
        # thin bottom divider
        header.add_widget(Label(
            text="[b]Offline RAG[/b]", markup=True,
            color=_TEXT_WHITE, font_size=sp(16),
            halign="center", valign="middle",
        ))
        root.add_widget(header)

        # thin separator line
        sep = Widget(size_hint=(1, None), height=dp(1))
        _paint_bg(sep, _DIVIDER)
        root.add_widget(sep)

        # ── Message list ──────────────────────────────────────────── #
        self._scroll = ScrollView(
            size_hint=(1, 1),
            do_scroll_x=False,
            bar_width=dp(3),
            scroll_type=["bars", "content"],
        )
        _paint_bg(self._scroll, _BG)

        self._msg_list = BoxLayout(
            orientation="vertical",
            size_hint=(1, None),
            spacing=0,
        )
        self._msg_list.bind(minimum_height=self._msg_list.setter("height"))
        self._scroll.add_widget(self._msg_list)
        root.add_widget(self._scroll)

        # Welcome message
        self._add_row(
            "Hi! I'm your offline assistant.\n"
            "Add documents in the [b]Docs[/b] tab, then ask me anything.",
            role="assistant",
        )

        # ── Input bar ─────────────────────────────────────────────── #
        # Outer container gives the bottom-bar its dark background
        outer = BoxLayout(
            size_hint=(1, None), height=dp(68),
            padding=[dp(12), dp(8), dp(12), dp(8)],
            spacing=dp(8),
        )
        _paint_bg(outer, _HEADER_BG)

        # Rounded text field container
        field_wrap = BoxLayout(
            size_hint=(1, 1),
            padding=[dp(14), dp(6), dp(50), dp(6)],  # leave room for send btn
        )
        _paint_bg(field_wrap, _INPUT_BG, radius=22)

        self._input = TextInput(
            hint_text        = "Message Offline RAG…",
            multiline        = False,
            size_hint        = (1, 1),
            font_size        = sp(14),
            foreground_color = _TEXT_WHITE,
            hint_text_color  = _TEXT_MUTED,
            background_color = (0, 0, 0, 0),   # transparent — field_wrap draws bg
            cursor_color     = _TEXT_WHITE,
            padding          = [0, dp(4)],
        )
        self._input.bind(on_text_validate=self._on_send)
        field_wrap.add_widget(self._input)

        # Floating send button (arrow ↑) overlaid on the right
        send_anchor = AnchorLayout(
            size_hint=(None, 1), width=dp(56),
            anchor_x="center", anchor_y="center",
        )
        send_btn = Button(
            text             = "↑",
            size_hint        = (None, None),
            size             = (dp(36), dp(36)),
            font_size        = sp(18),
            bold             = True,
            background_normal= "",
            background_color = _SEND_BG,
            color            = _TEXT_WHITE,
        )
        # round button via canvas
        with send_btn.canvas.before:
            Color(*_SEND_BG)
            self._send_rect = RoundedRectangle(radius=[dp(18)])
        send_btn.bind(
            pos =lambda w, _: setattr(self._send_rect, "pos",  w.pos),
            size=lambda w, _: setattr(self._send_rect, "size", w.size),
        )
        send_btn.bind(on_release=self._on_send)
        send_anchor.add_widget(send_btn)

        outer.add_widget(field_wrap)
        outer.add_widget(send_anchor)
        root.add_widget(outer)

        self.add_widget(root)

    # ── helpers ───────────────────────────────────────────────────── #

    def _add_row(self, text: str, role: str) -> MessageRow:
        row = MessageRow(text, role=role)
        self._msg_list.add_widget(row)
        Clock.schedule_once(lambda *_: self._scroll_down(), 0.05)
        return row

    def _scroll_down(self):
        self._scroll.scroll_y = 0

    def _show_typing(self):
        self._typing_indicator = _TypingIndicator()
        self._msg_list.add_widget(self._typing_indicator)
        Clock.schedule_once(lambda *_: self._scroll_down(), 0.05)

    def _hide_typing(self):
        if self._typing_indicator:
            self._typing_indicator.stop()
            self._msg_list.remove_widget(self._typing_indicator)
            self._typing_indicator = None

    # ── events ────────────────────────────────────────────────────── #

    def _on_send(self, *_):
        question = self._input.text.strip()
        if not question:
            return
        self._input.text = ""
        self._add_row(question, role="user")
        self._show_typing()

        from src.rag.pipeline import ask
        ask(
            question,
            top_k=4,
            stream_cb=self._on_token,
            on_done=self._on_done,
        )

    @mainthread
    def _on_token(self, token: str):
        if self._typing_indicator:
            self._hide_typing()
            self._current_row = self._add_row("", role="assistant")
        if self._current_row:
            self._current_row.append(token)
            Clock.schedule_once(lambda *_: self._scroll_down(), 0.02)

    @mainthread
    def _on_done(self, success: bool, message: str):
        self._hide_typing()
        if not success:
            if self._current_row:
                self._current_row._label.text = f"[color=ff5555]{message}[/color]"
            else:
                self._add_row(message, role="system")
        self._current_row = None
