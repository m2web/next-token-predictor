
import tkinter as tk
from tkinter import messagebox

# --- Imports and API Key Handling ---
import openai
import os

def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY environment variable. Please set your OpenAI API key.")
    return openai.OpenAI(api_key=api_key)

client = get_openai_client()

class TokenPredictorApp:
    def __init__(self, root):
        """Initialize the Tkinter GUI and set up widgets."""
        self.root = root
        self.root.title("Next Token Predictor")
        self.temperature = tk.DoubleVar(value=0.2)

        # Title
        tk.Label(root, text="Next Token Predictor", font=("Helvetica", 16, "bold")).pack(pady=(10, 0))

        # Frame for prompt and predictions
        frame_top = tk.Frame(root)
        frame_top.pack(pady=10, padx=10, fill=tk.X)

        # Prompt Entry
        prompt_outer = tk.Frame(frame_top, bg="#f8f8ff")
        prompt_outer.pack(side=tk.LEFT, padx=(0, 20), fill=tk.Y)
        tk.Label(prompt_outer, text="Enter your prompt here:", font=("Arial", 11, "bold"), bg="#f8f8ff").pack(anchor="w", padx=2, pady=(10, 0))
        prompt_frame = tk.Frame(prompt_outer, bd=1, relief="solid", bg="#f8f8ff")
        prompt_frame.pack(fill=tk.X, padx=0, pady=(0,0))
        self.prompt_entry = tk.Text(prompt_frame, width=50, height=5, font=("Arial", 12), borderwidth=0, bg="#f8f8ff", wrap="word")
        self.prompt_entry.pack(padx=10, pady=10, fill=tk.X, expand=True)
        self.prompt_entry.insert("1.0", "It's a lovely day let's go to the ")
        self.debounce_id = None
        self.prompt_entry.bind("<KeyRelease>", self.on_prompt_keyrelease)

        # Predictions display
        prediction_frame = tk.Frame(frame_top, bd=1, relief="solid", bg="#e0ecff")
        prediction_frame.pack(side=tk.RIGHT)
        tk.Label(prediction_frame, text="Next Token Probability", bg="#e0ecff", font=("Arial", 13, "bold")).pack(anchor="w", padx=10, pady=(5, 0))
        self.prediction_labels = []
        for _ in range(10):
            label = tk.Label(prediction_frame, text="", anchor="w", bg="#e0ecff", cursor="hand2", font=("Arial", 12, "bold"))
            label.pack(anchor="w", padx=10)
            label.bind("<Button-1>", self.complete_prompt)
            self.prediction_labels.append(label)

        # Output bubble
        self.output_label = tk.Label(root, text="", font=("Arial", 12), wraplength=600,
                                     bg="#ffe4b5", bd=1, relief="solid", padx=10, pady=10)
        self.output_label.pack(pady=(10, 0), padx=20, fill=tk.X)

        # Temperature control
        temp_frame = tk.Frame(root)
        temp_frame.pack(pady=10)
        tk.Label(temp_frame, text="Model Temperature", font=("Arial", 12, "bold")).pack(side=tk.LEFT, padx=(10, 5))
        self.temp_entry = tk.Entry(temp_frame, width=5, textvariable=self.temperature, font=("Arial", 12, "bold"))
        self.temp_entry.pack(side=tk.LEFT)
        self.temp_entry.bind("<Return>", lambda event: self.update_predictions())

        # Trigger first prediction
        self.update_predictions()

    def on_prompt_keyrelease(self, event):
        if self.debounce_id is not None:
            self.root.after_cancel(self.debounce_id)
        self.debounce_id = self.root.after(1050, self.update_predictions)


    def predict_next_token(self, prompt, temperature):
        """
        Modularized prediction logic for easier testing.
        Returns: (completed_text, sorted_tokens)
        """
        response = client.completions.create(
            model="gpt-3.5-turbo-instruct",
            prompt=prompt,
            max_tokens=1,
            logprobs=5,
            temperature=temperature
        )
        top_tokens = response.choices[0].logprobs.top_logprobs[0]
        best_token = max(top_tokens, key=top_tokens.get)
        completed = prompt + best_token
        sorted_tokens = sorted(top_tokens.items(), key=lambda x: -x[1])[:10]
        return completed, sorted_tokens

    def update_predictions(self):
        """
        Get the prompt, run prediction, and update the GUI display.
        """
        prompt = self.prompt_entry.get("1.0", tk.END).strip()
        if not prompt:
            for lbl in self.prediction_labels:
                lbl.config(text="", fg="black")
            self.output_label.config(text="")
            return

        try:
            temp = float(self.temp_entry.get())
            completed, sorted_tokens = self.predict_next_token(prompt, temp)
            self.output_label.config(text=completed)

            # Display top tokens and probabilities
            for i, (token, logprob) in enumerate(sorted_tokens):
                prob = round(100 * (10 ** logprob), 2)
                # Make whitespace tokens visible
                display_token = token
                if token == " ":
                    display_token = "[space]"
                elif token == "\t":
                    display_token = "[tab]"
                elif token == "\n":
                    display_token = "[newline]"
                elif token.strip() == "":
                    display_token = repr(token)
                self.prediction_labels[i].config(text=f"{display_token}  ({prob}%)", fg="blue")
                self.prediction_labels[i].token = token  # attach for click event
            for j in range(i + 1, 10):
                self.prediction_labels[j].config(text="", fg="black")
                self.prediction_labels[j].token = None
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def complete_prompt(self, event):
        token = event.widget.token
        if token:
            current = self.prompt_entry.get("1.0", tk.END)
            # If the prompt already ends with a space and the token starts with a space, avoid double space
            if current.endswith(" ") and token.startswith(" "):
                self.prompt_entry.insert(tk.END, token.lstrip())
            else:
                self.prompt_entry.insert(tk.END, token)
            self.update_predictions()

if __name__ == "__main__":
    root = tk.Tk()
    app = TokenPredictorApp(root)
    root.mainloop()
