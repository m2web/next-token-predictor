import tkinter as tk

root = tk.Tk()
root.title("Chat Bubble Example")
root.configure(bg="#f1f1f1")

def add_chat_bubble(text, side="user"):
    bubble_frame = tk.Frame(chat_frame, bg="#f1f1f1")

    if side == "user":
        bubble = tk.Label(
            bubble_frame,
            text=text,
            bg="#dcf8c6",  # Light green like WhatsApp
            fg="black",
            wraplength=300,
            justify="left",
            padx=10,
            pady=6,
            font=("Arial", 11),
            relief=tk.FLAT,
            bd=1
        )
        bubble.pack(anchor='e', padx=5, pady=2)
    else:
        bubble = tk.Label(
            bubble_frame,
            text=text,
            bg="white",
            fg="black",
            wraplength=300,
            justify="left",
            padx=10,
            pady=6,
            font=("Arial", 11),
            relief=tk.FLAT,
            bd=1
        )
        bubble.pack(anchor='w', padx=5, pady=2)

    bubble_frame.pack(fill='x', anchor='w' if side == 'bot' else 'e', pady=2)

# Chat area
chat_frame = tk.Frame(root, bg="#f1f1f1")
chat_frame.pack(fill='both', expand=True, padx=10, pady=10)

# Sample messages
add_chat_bubble("complete this sentence – It’s a beautiful day, let's go to the", side="user")
add_chat_bubble("It's a beautiful day, let's go to the park.", side="bot")

root.mainloop()
