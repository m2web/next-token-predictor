# Next Token Predictor

🚀 **Next Token Predictor**

A simple, interactive Python tool for predicting the next token in a text sequence using OpenAI's GPT model. This project is designed for educational purposes, NLP experimentation, and as a potential foundation for more advanced AI projects.

![App UI](assets/final_next_token_predictor_ui.png)

## Features

- 🧠 Predicts the next token in a user-provided text sequence
- Uses OpenAI's GPT model to predict the next token in your sequence
- Clean, user-friendly interface (see screenshot above)
- Easy to extend and customize for your own experiments

## Getting Started

### Note on Language Model Predictions

Unlike querying a database for specific records, using Large Language Models (LLMs) is fundamentally probabilistic. The model predicts the next token based on probability distributions learned from vast amounts of data—not with certainty or factual lookup. This means the results are the most likely continuations, not guaranteed facts.

That said, I’m thrilled that technology has advanced to this point! 😊 LLMs will continue to open up new possibilities for creativity and innovation—not only in software development, but also in project management, data modeling, marketing, medicine, and more. The potential applications are vast, even though the results remain probabilistic and are based on learned patterns rather than direct data retrieval.

### Prerequisites

- Python 3.8+
- To use the Streamlit web app, you will also need to install Streamlit:

   ```sh
   pip install streamlit
   ```

### Installation

1. Clone this repository:

   ```sh
   git clone https://github.com/m2web/next-token-predictor.git
   cd next-token-predictor
   ```

2. (Optional) Create and activate a virtual environment:

   ```sh
   python -m venv venv
   .\venv\Scripts\activate
   ```

## How to Use

### 1. Jupyter Notebook (`logprobs.ipynb`)

- **Purpose:** Interactive code exploration and analysis.
- **How to use:**
   1. Open the notebook in Jupyter (VS Code, JupyterLab, or classic Jupyter).
   2. Run each cell sequentially from top to bottom.
   3. Follow the instructions in markdown cells and modify code cells as needed.
   4. View token probabilities and experiment with prompts and parameters.

### 2. Streamlit Web App (`next_token_predictor_streamlit.py`)

- **Purpose:** Web-based, interactive next-token prediction and diagnostics.
- **How to use:**
   1. Open a terminal in the project folder.
   2. Run:

       ```sh
       streamlit run next_token_predictor_streamlit.py
       ```

   3. The app will open in your browser.
   4. Enter your prompt, adjust parameters, and view top token predictions, probability tables, and diagnostics.

### 3. Tkinter Desktop App (`next_token_predictor.py`)

- **Purpose:** Simple desktop GUI for quick next-token prediction.
- **How to use:**
   1. Open a terminal in the project folder.
   2. Run:

      ```python
       python next_token_predictor.py
       ```

   3. A windowed app will appear.
   4. Enter your prompt, adjust temperature, and view top token predictions. Click tokens to autocomplete your prompt.

All three require a valid OpenAI API key set in your environment. Choose the interface that best fits your workflow: notebook for code, Streamlit for web, Tkinter for desktop.

## Project Structure

```text
next-token-predictor/
├── assets/
│   ├── final_next_token_predictor_ui.png      # UI screenshot
│   └── token-predictor.drawio.pdf             # Architecture diagram
├── logprobs.ipynb                             # Jupyter notebook for code-first exploration
├── next_token_predictor.py                    # Tkinter desktop app
├── next_token_predictor_streamlit.py          # Streamlit web app
├── README.md                                  # Project documentation
├── requirements.txt                           # Python dependencies
├── .gitignore                                 # Git ignore file
```

## Application Variants: Notebook, Streamlit, and Tkinter

This project provides three main ways to experiment with next-token prediction:

### 1. Jupyter Notebook (via `logprobs.ipynb`)

- **Purpose:** Interactive, code-first exploration of next-token probabilities.
- **Features:**
- Step-by-step code cells to query OpenAI models and display token probabilities.
- Visualizes results using pandas DataFrames.
- Ideal for data scientists and developers who want to tinker, analyze, and extend the logic.
- **Purpose:** User-friendly web interface for next-token prediction and diagnostics.
- **Features:**
- Modern UI with sliders, tables, and infographics.
- Shows top-10 next tokens, probability distributions, and model diagnostics (entropy, perplexity, latency).
- Allows interactive parameter tuning (temperature, top-p, etc.).
- No coding required for users—just run and interact.
- **Usage:** Run with `streamlit run next_token_predictor_streamlit.py` and use in your browser.

### 2. Tkinter Desktop App (`next_token_predictor.py`)

- **Purpose:** Simple desktop GUI for quick next-token prediction.
- **Features:**
- Click tokens to autocomplete your prompt.
- Lightweight and easy to run on most systems.
- **Usage:** Run with `python next_token_predictor.py` to launch the desktop app.

---

### Similarities

- All variants use OpenAI’s GPT models to predict the next token.
- Each displays the top predicted tokens and their probabilities.
- **Streamlit App:** Best for interactive web-based use, visualization, and sharing with non-coders.
- **Tkinter App:** Best for quick desktop use without a browser or notebook environment.

## Contributing

## License

## Acknowledgements

- Inspired by classic NLP and AI research

---

🌟 _If you find this project useful, please star the repo!_
