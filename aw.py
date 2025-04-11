import tkinter as tk
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import threading

# Load DialoGPT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

# Global chat history
chat_history_ids = None

# Function to generate bot response
def get_response(user_input):
    global chat_history_ids
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
    bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1) if chat_history_ids is not None else new_input_ids
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    bot_response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return bot_response

# Tkinter GUI
class ChatApp:
    def _init_(self, root):
        self.root = root
        self.root.title("AI Chatbot")
        self.root.geometry("500x600")

        self.chat_area = tk.Text(root, wrap=tk.WORD, font=("Arial", 12))
        self.chat_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.chat_area.config(state=tk.DISABLED)

        self.entry = tk.Entry(root, font=("Arial", 12))
        self.entry.pack(padx=10, pady=10, fill=tk.X)
        self.entry.bind("<Return>", self.send_message)

    def send_message(self, event=None):
        user_input = self.entry.get()
        self.entry.delete(0, tk.END)
        self.display_message("You", user_input)

        threading.Thread(target=self.generate_reply, args=(user_input,)).start()

    def display_message(self, sender, message):
        self.chat_area.config(state=tk.NORMAL)
        self.chat_area.insert(tk.END, f"{sender}: {message}\n")
        self.chat_area.yview(tk.END)
        self.chat_area.config(state=tk.DISABLED)

    def generate_reply(self, user_input):
        if user_input.lower() == "bye":
            self.display_message("Bot", "Goodbye!")
            return
        bot_reply = get_response(user_input)
        self.display_message("Bot", bot_reply)

# Run the app
if _name_ == "_main_":
    root = tk.Tk()
    app = ChatApp(root)
    root.mainloop()