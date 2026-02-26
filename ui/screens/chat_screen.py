"""
chat_screen.py — Unified single-screen chat + document interface.

Design:
  • Header: "Offline RAG" title only — no tabs or mode toggles.
  • Chat area inheriting ChatGPT dark style.
  • Bottom bar: [+] attach  |  [text input pill]  |  [↑ send]
  • Tap + to pick a PDF/TXT via the native file browser.
    - Document ingestion progress shown inline as a status card.
    - Once any doc is loaded the AI auto-answers from it (RAG mode).
    - With no docs, the AI just chats freely (direct mode).
  • Model loading / extraction progress shown in the welcome message.
"""
from __future__ import annotations

from kivy.uix.screenmanager import Screen
from kivy.uix.boxlayout     import BoxLayout
from kivy.uix.anchorlayout  import AnchorLayout
from kivy.uix.scrollview    import ScrollView
from kivy.uix.label         import Label
from kivy.uix.textinput     import TextInput
from kivy.uix.button        import Button
from kivy.uix.widget        import Widget
from kivy.uix.progressbar   import ProgressBar
from kivy.clock             import Clock, mainthread
from kivy.metrics           import dp, sp
from kivy.graphics          import Color, RoundedRectangle, Rectangle
from kivy.animation         import Animation

# ── Palette ───────────────────────────────────────────────────────── #
_BG        = (0.102, 0.102, 0.102, 1)   # #1a1a1a  page background
_HDR_BG    = (0.078, 0.078, 0.078, 1)   # #141414  header strip
_USER_BG   = (0.184, 0.184, 0.184, 1)   # #2f2f2f  user bubble
_INPUT_BG  = (0.173, 0.173, 0.173, 1)   # #2c2c2c  text-input wrap
_GREEN     = (0.098, 0.761, 0.490, 1)   # #19c37d  ChatGPT green
_ADD_BG    = (0.220, 0.220, 0.220, 1)   # #383838  + button
_WHITE     = (1,    1,    1,    1)
_MUTED     = (0.55, 0.55, 0.58, 1)
_DIVIDER   = (0.20, 0.20, 0.20, 1)
_DOC_CARD  = (0.12, 0.22, 0.17, 1)      # dark teal for doc status card
_ATTACH_BG = (0.165, 0.165, 0.165, 1)   # attachment preview card background
_RED_ICON  = (0.85, 0.18, 0.18, 1)      # PDF icon red


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #

def _paint(widget, color, radius: float = 0):
    """Bind a solid colour background to a widget's canvas.before."""
    with widget.canvas.before:
        Color(*color)
        r = (RoundedRectangle(radius=[dp(radius)]) if radius else Rectangle())
    widget.bind(
        pos =lambda w, _: setattr(r, "pos",  w.pos),
        size=lambda w, _: setattr(r, "size", w.size),
    )
    return r


# ------------------------------------------------------------------ #
#  Avatar circle (letters "U" / "AI")                                 #
# ------------------------------------------------------------------ #

class _Avatar(Widget):
    _COLS = {
        "user":      (0.40, 0.40, 0.90, 1),
        "assistant": _GREEN,
        "system":    (0.80, 0.20, 0.20, 1),
    }

    def __init__(self, role: str, **kw):
        super().__init__(size_hint=(None, None), size=(dp(32), dp(32)), **kw)
        letter = {"user": "U", "assistant": "AI", "system": "!"}.get(role, "?")
        with self.canvas:
            Color(*self._COLS.get(role, (0.5, 0.5, 0.5, 1)))
            self._circ = RoundedRectangle(radius=[dp(16)])
        self.bind(pos=self._upd, size=self._upd)
        self._lbl = Label(text=letter, font_size=sp(11), bold=True, color=_WHITE)
        self.add_widget(self._lbl)

    def _upd(self, *_):
        self._circ.pos  = self.pos
        self._circ.size = self.size
        self._lbl.center = self.center


# ------------------------------------------------------------------ #
#  Message row (user bubble right / assistant text left)              #
# ------------------------------------------------------------------ #

class MessageRow(BoxLayout):
    def __init__(self, text: str, role: str = "assistant", **kw):
        super().__init__(
            orientation="horizontal",
            size_hint=(1, None),
            padding=[dp(12), dp(8), dp(12), dp(8)],
            spacing=dp(10),
            **kw,
        )
        self.role = role
        self._lbl = Label(
            text=text, markup=True,
            size_hint_y=None, text_size=(None, None),
            halign="left", valign="top",
            color=_WHITE, font_size=sp(14.5),
        )
        self._lbl.bind(texture_size=self._on_tex)
        self.bind(width=self._on_w)
        _paint(self, _BG)

        if role == "user":
            self._build_user()
        else:
            self._build_asst()

    def _build_user(self):
        self.add_widget(Widget(size_hint_x=1))           # push right
        bub = BoxLayout(size_hint=(None, None), padding=[dp(12), dp(10)])
        _paint(bub, _USER_BG, radius=18)
        bub.add_widget(self._lbl)
        self._bub = bub
        self.add_widget(bub)
        self.add_widget(_Avatar("user"))

    def _build_asst(self):
        self.add_widget(_Avatar("assistant"))
        self.add_widget(self._lbl)

    def _on_tex(self, lbl, ts):
        lbl.height    = ts[1] + dp(4)
        lbl.text_size = (lbl.width or 1, None)
        if self.role == "user" and hasattr(self, "_bub"):
            self._bub.width  = min(ts[0] + dp(28), self.width * 0.82)
            self._bub.height = lbl.height + dp(20)
        self.height = max(lbl.height + dp(20), dp(52))

    def _on_w(self, *_):
        avail = self.width - dp(72)
        if self.role == "user":
            self._lbl.text_size = (avail * 0.82, None)
        else:
            self._lbl.text_size = (avail, None)

    def append(self, token: str):
        self._lbl.text += token


# ------------------------------------------------------------------ #
#  Attachment preview card (ChatGPT-style, shown above input bar)     #
# ------------------------------------------------------------------ #

class AttachmentPreviewCard(BoxLayout):
    """
    Shows a PDF/TXT attachment thumbnail above the message input,
    matching the ChatGPT attachment card style.
    """
    def __init__(self, filepath: str, on_remove, **kw):
        import os
        super().__init__(
            orientation="horizontal",
            size_hint=(None, None),
            size=(dp(220), dp(68)),
            padding=[dp(10), dp(8), dp(8), dp(8)],
            spacing=dp(10),
            **kw,
        )
        _paint(self, _ATTACH_BG, radius=14)

        fname = os.path.basename(filepath)
        ext   = os.path.splitext(fname)[1].upper().lstrip(".") or "FILE"
        try:
            sz_kb = os.path.getsize(filepath) // 1024
            size_txt = f"{sz_kb} KB" if sz_kb < 1024 else f"{sz_kb//1024} MB"
        except Exception:
            size_txt = ""

        # ── PDF/TXT icon box ────────────────────────────────────────── #
        icon_box = BoxLayout(
            size_hint=(None, None), size=(dp(42), dp(42)),
        )
        _paint(icon_box, _RED_ICON, radius=8)
        icon_lbl = Label(
            text=f"[b]{ext[:4]}[/b]", markup=True,
            font_size=sp(10), color=_WHITE,
            halign="center", valign="middle",
        )
        icon_lbl.bind(size=lambda w, _: setattr(w, "text_size", w.size))
        icon_box.add_widget(icon_lbl)
        self.add_widget(icon_box)

        # ── Filename + size ─────────────────────────────────────────── #
        info = BoxLayout(
            orientation="vertical", size_hint=(1, 1), spacing=dp(2),
        )
        # Truncate long filenames
        display = fname if len(fname) <= 22 else fname[:19] + "…"
        name_lbl = Label(
            text=f"[b]{display}[/b]", markup=True,
            font_size=sp(12), color=_WHITE,
            halign="left", valign="bottom",
            size_hint_y=None, height=dp(22),
        )
        name_lbl.bind(size=lambda w, _: setattr(w, "text_size", (w.width, None)))

        type_lbl = Label(
            text=f"{ext} · {size_txt}" if size_txt else ext,
            font_size=sp(10.5), color=_MUTED,
            halign="left", valign="top",
            size_hint_y=None, height=dp(18),
        )
        type_lbl.bind(size=lambda w, _: setattr(w, "text_size", (w.width, None)))

        info.add_widget(name_lbl)
        info.add_widget(type_lbl)
        self.add_widget(info)

        # ── × remove button ─────────────────────────────────────────── #
        x_btn = Button(
            text="✕", font_size=sp(13),
            size_hint=(None, None), size=(dp(24), dp(24)),
            background_normal="", background_color=(0, 0, 0, 0),
            color=_MUTED,
        )
        x_btn.bind(on_release=lambda *_: on_remove())
        anc = AnchorLayout(
            size_hint=(None, 1), width=dp(28),
            anchor_x="center", anchor_y="top",
        )
        anc.add_widget(x_btn)
        self.add_widget(anc)


# ------------------------------------------------------------------ #
#  Document ingestion status card                                      #
# ------------------------------------------------------------------ #

class DocStatusCard(BoxLayout):
    """
    Inline card that shows file name + progress indicator while a
    document is being chunked and indexed.
    """
    def __init__(self, filename: str, **kw):
        super().__init__(
            orientation="vertical",
            size_hint=(1, None),
            padding=[dp(14), dp(8), dp(14), dp(8)],
            spacing=dp(4),
            **kw,
        )
        _paint(self, _BG)

        inner = BoxLayout(
            orientation="vertical",
            size_hint=(1, None),
            padding=[dp(14), dp(12)],
            spacing=dp(6),
        )
        _paint(inner, _DOC_CARD, radius=14)

        self._title = Label(
            text=f"[b]📄  {filename}[/b]",
            markup=True,
            color=_WHITE, font_size=sp(13),
            size_hint_y=None, height=dp(22),
            halign="left", valign="middle",
        )
        self._title.bind(size=lambda w, _: setattr(w, "text_size", (w.width, None)))

        self._status = Label(
            text="Indexing…",
            color=_GREEN, font_size=sp(12),
            size_hint_y=None, height=dp(18),
            halign="left", valign="middle",
        )
        self._status.bind(size=lambda w, _: setattr(w, "text_size", (w.width, None)))

        self._bar = ProgressBar(
            max=100, value=10,
            size_hint=(1, None), height=dp(5),
        )

        inner.add_widget(self._title)
        inner.add_widget(self._status)
        inner.add_widget(self._bar)
        inner.bind(minimum_height=inner.setter("height"))

        self.add_widget(inner)
        self.bind(minimum_height=self.setter("height"))

    def set_done(self, success: bool, message: str):
        self._bar.value = 100 if success else 0
        col  = "00cc66" if success else "ff5555"
        icon = "✅" if success else "❌"
        self._status.text   = f"[color={col}]{icon}  {message}[/color]"
        self._status.markup = True


# ------------------------------------------------------------------ #
#  Typing indicator  ● ● ●                                            #
# ------------------------------------------------------------------ #

class _TypingIndicator(BoxLayout):
    def __init__(self, **kw):
        super().__init__(
            orientation="horizontal",
            size_hint=(1, None), height=dp(40),
            padding=[dp(56), dp(4)], spacing=dp(6),
            **kw,
        )
        self._dots: list[Label] = []
        for _ in range(3):
            d = Label(
                text="●", font_size=sp(10), color=_MUTED,
                size_hint=(None, None), size=(dp(14), dp(14)),
            )
            self._dots.append(d)
            self.add_widget(d)
        self._tick = 0
        Clock.schedule_interval(self._anim, 0.42)

    def _anim(self, *_):
        for i, d in enumerate(self._dots):
            d.color = _WHITE if i == self._tick % 3 else _MUTED
        self._tick += 1

    def stop(self):
        Clock.unschedule(self._anim)


# ================================================================== #
#  ChatScreen                                                         #
# ================================================================== #

class ChatScreen(Screen):
    """
    Single-screen UI.  No tab bar.
    Internal mode tracked automatically:
      _has_docs=True  →  RAG (answer from indexed document chunks)
      _has_docs=False →  direct LLM chat with rolling history
    """

    def __init__(self, **kw):
        super().__init__(**kw)
        self._history:        list                    = []
        self._pending_q:      str                     = ""
        self._current_row:    MessageRow | None       = None
        self._typing:         _TypingIndicator | None = None
        self._has_docs:       bool                    = False
        self._pending_attach: str | None              = None   # path of staged file
        self._attach_card:    AttachmentPreviewCard | None = None
        self._scroll_pending: bool                   = False   # debounce auto-scroll
        self._build_ui()

    # ---------------------------------------------------------------- #
    #  Layout                                                           #
    # ---------------------------------------------------------------- #

    def _build_ui(self):
        root = BoxLayout(orientation="vertical")
        _paint(root, _BG)

        # ── Header ──────────────────────────────────────────────────── #
        hdr = BoxLayout(
            size_hint=(1, None), height=dp(54),
            padding=[dp(16), dp(0)],
        )
        _paint(hdr, _HDR_BG)
        hdr.add_widget(Label(
            text="[b]O-RAG[/b]", markup=True,
            color=_WHITE, font_size=sp(16),
            halign="center", valign="middle",
        ))
        root.add_widget(hdr)

        sep = Widget(size_hint=(1, None), height=dp(1))
        _paint(sep, _DIVIDER)
        root.add_widget(sep)

        # ── Message list ─────────────────────────────────────────────── #
        self._scroll = ScrollView(
            size_hint=(1, 1), do_scroll_x=False, bar_width=dp(3),
        )
        _paint(self._scroll, _BG)

        self._msgs = BoxLayout(
            orientation="vertical",
            size_hint=(1, None), spacing=0,
        )
        self._msgs.bind(minimum_height=self._msgs.setter("height"))
        self._scroll.add_widget(self._msgs)
        root.add_widget(self._scroll)

        # Welcome message — text updated when model is ready
        self._welcome = self._add_msg(
            "Hello! I'm your offline AI assistant.\n\n"
            "⏳  [b]Preparing model…[/b] This may take a moment on first launch.",
            role="assistant",
        )

        # ── Input area (attachment strip + bar) ─────────────────────── #
        input_area = BoxLayout(
            orientation="vertical",
            size_hint=(1, None), height=dp(74),
        )
        _paint(input_area, _HDR_BG)

        # Attachment preview strip — hidden until a file is picked
        self._attach_strip = BoxLayout(
            orientation="horizontal",
            size_hint=(1, None), height=0,
            padding=[dp(10), dp(6), dp(10), dp(0)],
        )
        _paint(self._attach_strip, _HDR_BG)
        input_area.add_widget(self._attach_strip)

        bar = BoxLayout(
            size_hint=(1, None), height=dp(74),
            padding=[dp(10), dp(8), dp(10), dp(8)],
            spacing=dp(8),
        )

        # [+] attach button
        add_btn = Button(
            text="+",
            font_size=sp(26), bold=True,
            size_hint=(None, None), size=(dp(48), dp(48)),
            background_normal="", background_color=(0, 0, 0, 0),
            color=_WHITE,
        )
        _paint(add_btn, _ADD_BG, radius=24)
        add_btn.bind(on_release=self._on_attach)
        bar.add_widget(add_btn)

        # Text input pill
        pill = BoxLayout(
            size_hint=(1, 1),
            padding=[dp(14), dp(8), dp(52), dp(8)],
        )
        _paint(pill, _INPUT_BG, radius=22)

        self._input = TextInput(
            hint_text="Message…",
            multiline=False, size_hint=(1, 1),
            font_size=sp(14.5),
            foreground_color=_WHITE,
            hint_text_color=_MUTED,
            background_color=(0, 0, 0, 0),
            cursor_color=_WHITE,
            padding=[0, dp(4)],
        )
        self._input.bind(on_text_validate=self._on_send)
        pill.add_widget(self._input)

        # [↑] send button overlaid on pill right
        send_anc = AnchorLayout(
            size_hint=(None, 1), width=dp(52),
            anchor_x="center", anchor_y="center",
        )
        send_btn = Button(
            text="↑", font_size=sp(20), bold=True,
            size_hint=(None, None), size=(dp(40), dp(40)),
            background_normal="", background_color=(0, 0, 0, 0),
            color=_WHITE,
        )
        _paint(send_btn, _GREEN, radius=20)
        send_btn.bind(on_release=self._on_send)
        send_anc.add_widget(send_btn)

        bar.add_widget(pill)
        bar.add_widget(send_anc)
        input_area.add_widget(bar)
        root.add_widget(input_area)

        self.add_widget(root)

        # Register model-ready callbacks once pipeline is up
        Clock.schedule_once(self._register_pipeline_callbacks, 0.8)

    # ---------------------------------------------------------------- #
    #  Model progress / ready callbacks                                 #
    # ---------------------------------------------------------------- #

    def _register_pipeline_callbacks(self, *_):
        from rag.pipeline import register_auto_download_callbacks
        register_auto_download_callbacks(
            on_progress=self._on_model_progress,
            on_done    =self._on_model_ready,
        )

    @mainthread
    def _on_model_progress(self, frac: float, text: str):
        pct     = int(frac * 100)
        filled  = "█" * (pct // 10)
        empty   = "░" * (10 - pct // 10)
        self._welcome._lbl.text = (
            f"⏳  [b]Preparing model…[/b]  {pct}%\n"
            f"[size=11sp][color=888888]{filled}{empty}[/color][/size]\n"
            f"{text}"
        )

    @mainthread
    def _on_model_ready(self, success: bool, message: str):
        if success:
            self._welcome._lbl.text = (
                "👋  [b]How can I assist you today?[/b]\n\n"
                "• Just type a message to chat with me.\n"
                "• Tap [b]＋[/b] to attach a [b]PDF[/b] or [b]TXT[/b] — "
                "I'll answer questions about its content."
            )
            self._request_storage_permissions()
        else:
            self._welcome._lbl.text = (
                f"[color=ff5555]⚠  Model failed to load:[/color]\n{message}\n\n"
                "Check your connection and restart the app."
            )

    # ---------------------------------------------------------------- #
    #  Storage permission request                                       #
    # ---------------------------------------------------------------- #

    def _request_storage_permissions(self, *_):
        """Ask for storage permissions on Android right after model loads."""
        import os
        if not os.environ.get("ANDROID_PRIVATE"):
            return  # desktop — no-op
        try:
            from android.permissions import request_permissions, Permission  # type: ignore
            sdk = 0
            try:
                from jnius import autoclass  # type: ignore
                sdk = autoclass("android.os.Build$VERSION").SDK_INT
            except Exception:
                pass
            if sdk >= 33:
                # Android 13+ — READ_MEDIA_IMAGES covers images;
                # documents/PDFs still come through SAF so no extra perm needed.
                request_permissions([
                    Permission.READ_MEDIA_IMAGES,
                    Permission.READ_MEDIA_VIDEO,
                ])
            else:
                request_permissions([
                    Permission.READ_EXTERNAL_STORAGE,
                    Permission.WRITE_EXTERNAL_STORAGE,
                ])
        except Exception as e:
            print(f"[permissions] Could not request: {e}")

    # ---------------------------------------------------------------- #
    #  Attach document via file picker                                  #
    # ---------------------------------------------------------------- #

    # Unique request code for startActivityForResult
    _PICK_REQ = 0x4F52   # "OR"

    def _on_attach(self, *_):
        # Guard: prevent double-open if picker is already visible
        if getattr(self, "_picker_open", False):
            return
        self._picker_open = True
        import os
        if os.environ.get("ANDROID_PRIVATE"):
            self._android_pick_file()
        else:
            self._desktop_pick_file()

    # -- Android path: native startActivityForResult ------------------

    def _android_pick_file(self):
        try:
            from jnius import autoclass          # type: ignore
            from android.activity import bind as activity_bind  # type: ignore

            # Register result handler before launching intent
            activity_bind(on_activity_result=self._on_activity_result)

            PythonActivity = autoclass("org.kivy.android.PythonActivity")
            Intent         = autoclass("android.content.Intent")

            intent = Intent(Intent.ACTION_GET_CONTENT)
            intent.setType("*/*")               # show all; filtered by MIME below
            intent.addCategory(Intent.CATEGORY_OPENABLE)

            # Restrict picker to PDF + plain-text via EXTRA_MIME_TYPES
            try:
                ArrayList = autoclass("java.util.ArrayList")
                mimes = ArrayList()
                mimes.add("application/pdf")
                mimes.add("text/plain")
                intent.putExtra("android.intent.extra.MIME_TYPES",
                                mimes.toArray())
            except Exception:
                # Fallback: accept everything; resolve_uri will validate ext
                pass

            PythonActivity.mActivity.startActivityForResult(
                intent, self._PICK_REQ
            )
        except Exception as e:
            import traceback; traceback.print_exc()
            self._picker_open = False
            self._add_msg(
                f"[color=ff5555]\u274c  Could not open file picker:[/color]\n{e}",
                role="assistant",
            )

    def _on_activity_result(self, request_code, result_code, data):
        # Unregister immediately so we don't receive stale callbacks
        try:
            from android.activity import unbind as activity_unbind  # type: ignore
            activity_unbind(on_activity_result=self._on_activity_result)
        except Exception:
            pass

        self._picker_open = False

        if request_code != self._PICK_REQ:
            return
        RESULT_OK = -1   # android.app.Activity.RESULT_OK
        if result_code != RESULT_OK or data is None:
            return

        try:
            uri = data.getData()
            if uri is None:
                return
            uri_str = uri.toString()
            Clock.schedule_once(lambda *_: self._process_picked_uri(uri_str), 0)
        except Exception as e:
            import traceback; traceback.print_exc()
            self._add_msg(
                f"[color=ff5555]\u274c  Could not read file URI:[/color]\n{e}",
                role="assistant",
            )

    @mainthread
    def _process_picked_uri(self, uri_str: str):
        try:
            from rag.chunker import resolve_uri
            path = resolve_uri(uri_str)
            self._stage_attachment(path)
        except Exception as e:
            import traceback; traceback.print_exc()
            self._add_msg(
                f"[color=ff5555]\u274c  Could not open file:[/color]\n{e}",
                role="assistant",
            )

    # -- Desktop path: plyer fallback ---------------------------------

    def _desktop_pick_file(self):
        try:
            from plyer import filechooser
            filechooser.open_file(
                on_selection=self._on_file_chosen,
                filters=[["Documents", "*.pdf", "*.txt", "*.PDF", "*.TXT"]],
                title="Pick a document",
                multiple=False,
            )
        except Exception as e:
            self._picker_open = False
            self._add_msg(
                "File picker unavailable on this device.\n"
                "Type the [b]full path[/b] to your file and send it — "
                f"e.g. [i]/sdcard/Download/report.pdf[/i]",
                role="assistant",
            )

    @mainthread
    def _on_file_chosen(self, selection):
        self._picker_open = False
        if not selection or selection[0] is None:
            return
        try:
            from rag.chunker import resolve_uri
            path = resolve_uri(selection[0])
            self._stage_attachment(path)
        except Exception as e:
            import traceback; traceback.print_exc()
            self._add_msg(
                f"[color=ff5555]\u274c  Could not open file:[/color]\n{e}",
                role="assistant",
            )

    def _stage_attachment(self, path: str):
        """Show the attachment preview card above the input bar."""
        import os
        self._pending_attach = path

        # Remove any existing card
        self._attach_strip.clear_widgets()

        card = AttachmentPreviewCard(
            filepath=path,
            on_remove=self._remove_attachment,
        )
        self._attach_card = card
        self._attach_strip.add_widget(card)

        # Expand the strip to show the card
        self._attach_strip.height = dp(80)
        # Grow the whole input_area
        self._attach_strip.parent.height = dp(154)

    @mainthread
    def _remove_attachment(self):
        """Dismiss the staged attachment card."""
        self._pending_attach = None
        self._attach_card    = None
        self._attach_strip.clear_widgets()
        self._attach_strip.height = 0
        self._attach_strip.parent.height = dp(74)

    def _start_ingest(self, path: str, fname: str):
        card = DocStatusCard(fname)
        self._msgs.add_widget(card)
        self._scroll_down()
        from rag.pipeline import ingest_document
        ingest_document(
            path,
            on_done=lambda ok, msg: self._ingest_done(card, ok, msg),
        )

    @mainthread
    def _ingest_done(self, card: DocStatusCard, ok: bool, msg: str):
        card.set_done(ok, msg)
        if ok:
            self._has_docs = True
            self._add_msg(
                "✅  Document indexed! Ask me anything about it.\n"
                "[color=888888][size=12sp]"
                "(Tap ＋ again to add more documents.)"
                "[/size][/color]",
                role="assistant",
            )
        else:
            self._add_msg(
                f"[color=ff5555]❌  Could not load document:[/color]\n{msg}",
                role="assistant",
            )
        self._scroll_down()

    # ---------------------------------------------------------------- #
    #  Handle plain file-path typed into chat                           #
    # ---------------------------------------------------------------- #

    def _maybe_load_path(self, text: str) -> bool:
        """If user pastes a file path, stage it as an attachment."""
        import os
        s = text.strip()
        is_path = (s.startswith("/") or (len(s) > 2 and s[1] == ":")) \
                  and os.path.isfile(s)
        if is_path:
            self._stage_attachment(s)
            return True
        return False

    # ---------------------------------------------------------------- #
    #  Send / receive                                                   #
    # ---------------------------------------------------------------- #

    def _on_send(self, *_):
        q    = self._input.text.strip()
        path = self._pending_attach

        # Nothing to do if both empty
        if not q and not path:
            return

        self._input.text = ""

        # If there is a staged file, ingest it first
        if path:
            import os
            fname = os.path.basename(path)
            self._remove_attachment()
            # Show a user bubble with the attachment + any typed text
            bubble_text = f"📎  [b]{fname}[/b]"
            if q:
                bubble_text += f"\n{q}"
            self._add_msg(bubble_text, role="user")
            self._start_ingest(path, fname)
            return

        # Plain text path typed into the input box
        if self._maybe_load_path(q):
            return

        self._pending_q = q
        self._add_msg(q, role="user")
        self._show_typing()

        if self._has_docs:
            from rag.pipeline import ask
            ask(q, top_k=4, stream_cb=self._on_token, on_done=self._on_done)
        else:
            from rag.pipeline import chat_direct
            chat_direct(
                q,
                history  =list(self._history),
                stream_cb=self._on_token,
                on_done  =self._on_done,
            )

    @mainthread
    def _on_token(self, token: str):
        if self._typing:
            self._hide_typing()
            self._current_row = self._add_msg("", role="assistant")
        if self._current_row:
            self._current_row.append(token)
            # Debounce: scroll at most once every 120 ms to stay smooth
            if not self._scroll_pending:
                self._scroll_pending = True
                Clock.schedule_once(self._do_scroll, 0.12)

    @mainthread
    def _on_done(self, success: bool, message: str):
        self._hide_typing()
        if success:
            if not self._has_docs and self._pending_q and self._current_row:
                # Strip Kivy markup tags before storing in history so the
                # raw text is sent to the model (markup tokens corrupt prompts)
                import re
                raw_ans = re.sub(r'\[/?[a-zA-Z][^\]]*\]', '',
                                 self._current_row._lbl.text).strip()
                self._history.append((self._pending_q, raw_ans))
                if len(self._history) > 10:
                    self._history = self._history[-10:]
        else:
            if self._current_row:
                self._current_row._lbl.text = (
                    f"[color=ff5555]{message}[/color]"
                )
            else:
                self._add_msg(message, role="assistant")
        self._pending_q   = ""
        self._current_row = None

    # ---------------------------------------------------------------- #
    #  Helpers                                                          #
    # ---------------------------------------------------------------- #

    def _add_msg(self, text: str, role: str = "assistant") -> MessageRow:
        row = MessageRow(text, role=role)
        self._msgs.add_widget(row)
        Clock.schedule_once(lambda *_: self._scroll_down(), 0.05)
        return row

    def _show_typing(self):
        self._typing = _TypingIndicator()
        self._msgs.add_widget(self._typing)
        Clock.schedule_once(lambda *_: self._scroll_down(), 0.05)

    def _hide_typing(self):
        if self._typing:
            self._typing.stop()
            self._msgs.remove_widget(self._typing)
            self._typing = None

    def _do_scroll(self, *_):
        """Debounced scroll — called at most every 120 ms during streaming."""
        self._scroll_pending = False
        self._scroll_down()

    def _scroll_down(self):
        """Smoothly animate to bottom of chat."""
        Animation.stop_all(self._scroll, "scroll_y")
        Animation(scroll_y=0, duration=0.15, t="out_quad").start(self._scroll)
