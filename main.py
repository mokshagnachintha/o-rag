import flet as ft
import time
import os
import threading

# Import the RAG backend logic validated in benchmarks
import sys
sys.path.insert(0, os.path.abspath('.'))
from rag.llm import LlamaCppModel, build_rag_prompt
from rag.retriever import HybridRetriever

# Model Paths determined best by benchmarks
QWEN_PATH = r"c:\Users\cmoks\Desktop\check\qwen2.5-1.5b-instruct-q4_k_m.gguf"
NOMIC_PATH = r"c:\Users\cmoks\Desktop\check\nomic-embed-text-v1.5.Q4_K_M.gguf"

class Message():
    def __init__(self, user_name: str, text: str, user_type: str):
        self.user_name = user_name
        self.text = text
        self.user_type = user_type

class ChatMessage(ft.Row):
    def __init__(self, message: Message):
        super().__init__()
        self.vertical_alignment = ft.CrossAxisAlignment.START
        self.controls = [
            ft.CircleAvatar(
                content=ft.Text(self.get_initials(message.user_name)),
                color=ft.colors.WHITE,
                bgcolor=self.get_avatar_color(message.user_name),
            ),
            ft.Column(
                [
                    ft.Text(message.user_name, weight="bold"),
                    ft.Text(message.text, selectable=True, width=ft.Window.width - 100 if ft.Window.width else 300),
                ],
                tight=True,
                spacing=5,
            ),
        ]

    def get_initials(self, user_name: str):
        return user_name[:1].capitalize()

    def get_avatar_color(self, user_name: str):
        if user_name == "User":
            return ft.colors.BLUE_GREY_400
        elif user_name == "System":
            return ft.colors.RED_400
        return ft.colors.BLUE_700

def main(page: ft.Page):
    page.title = "Mobile RAG - Qwen + Nomic"
    page.theme_mode = ft.ThemeMode.DARK
    page.padding = 10
    
    # RAG State
    chat_messages = ft.ListView(
        expand=True,
        spacing=10,
        auto_scroll=True,
    )
    
    llm = LlamaCppModel()
    retriever = HybridRetriever(alpha=0.5)
    
    status_text = ft.Text("Initializing Models...", size=12, color=ft.colors.GREY_500)
    
    def add_message(name, text, msg_type="bot"):
        m = Message(name, text, msg_type)
        chat_messages.controls.append(ChatMessage(m))
        page.update()

    def actually_load_models():
        try:
            add_message("System", "Loading Nomic Q4 Embedding Model...", "system")
            llm.load(NOMIC_PATH)
            retriever.reload()
            time.sleep(2) # Allow embeddings to cache
            add_message("System", "Embeddings Indexed. Loading Qwen 1.5B Generator...", "system")
            llm.unload()
            
            # Load generation model for persistent UI session
            llm.load(QWEN_PATH)
            add_message("System", "Ready! Models loaded completely locally.", "system")
            status_text.value = "🟢 Ready"
            page.update()
        except Exception as e:
            add_message("System", f"Critical Load Error: {e}", "system")
            status_text.value = "🔴 Error loading models"
            page.update()

    def load_models_async():
        add_message("System", "Verifying Model Files...", "system")
        
        # We need a progress updater string to show the user
        def update_dl_progress(frac: float, msg: str):
            status_text.value = f"Downloading: {msg}"
            page.update()
            
        def dl_finished(success: bool, msg: str):
            if success:
                actually_load_models()
            else:
                add_message("System", f"Failed to get models: {msg}", "system")
                status_text.value = "🔴 Download Failed"
                page.update()
                
        # Trigger the downloader from downloader.py
        from rag.downloader import auto_download_default
        auto_download_default(on_progress=update_dl_progress, on_done=dl_finished)

    # Start initialization checks in background
    threading.Thread(target=load_models_async, daemon=True).start()

    def send_click(e):
        if not new_message.value:
            return
            
        user_text = new_message.value
        new_message.value = ""
        new_message.focus()
        add_message("User", user_text, "user")
        
        status_text.value = "Searching documents..."
        page.update()
        
        # Async generation to not freeze UI
        def process_rag():
            try:
                # 1. Retrieve Context using Background Nomic Embeddings
                results = retriever.query(user_text, top_k=2)
                context_texts = [text for text, score in results]
                
                status_text.value = "Thinking..."
                page.update()
                
                # 2. Build Prompt and Generate using Qwen
                formatted_prompt = build_rag_prompt(context_texts, user_text)
                
                t0 = time.time()
                response = llm.generate(formatted_prompt, max_tokens=256, temperature=0.7)
                dt = time.time() - t0
                tok_sec = (len(response)/4.0) / dt if dt>0 else 0
                
                add_message("Qwen RAG", response, "bot")
                status_text.value = f"🟢 Ready ({tok_sec:.1f} t/s)"
                page.update()
                
            except Exception as e:
                add_message("System", f"Error generating text: {e}", "system")
                status_text.value = "🔴 Error"
                page.update()
                
        threading.Thread(target=process_rag, daemon=True).start()

    new_message = ft.TextField(
        hint_text="Ask about the Apollo landing context...",
        autofocus=True,
        shift_enter=True,
        min_lines=1,
        max_lines=5,
        filled=True,
        expand=True,
        on_submit=send_click,
    )

    page.add(
        ft.Row([ft.Text("⚡ Local Mobile RAG", weight="bold", size=20)], alignment=ft.MainAxisAlignment.CENTER),
        ft.Divider(),
        chat_messages,
        ft.Divider(),
        ft.Row([status_text], alignment=ft.MainAxisAlignment.START),
        ft.Row(
            [
                new_message,
                ft.IconButton(
                    icon=ft.icons.SEND_ROUNDED,
                    tooltip="Send message",
                    on_click=send_click,
                ),
            ]
        ),
    )

ft.app(target=main)
