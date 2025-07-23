
import streamlit as st
import openai
import os

# Set up OpenAI API key
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="Next Token Predictor", layout="centered")

st.title("🔮 Next Token Predictor")

# User input
prompt = st.text_input("Enter your prompt:", "It's a beautiful day, let's go to the")

temperature = st.slider("Model Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.05)

if prompt:
    try:
        response = client.completions.create(
            model="gpt-3.5-turbo-instruct",
            prompt=prompt,
            max_tokens=1,
            logprobs=10,
            temperature=temperature
        )

        top_tokens = response.choices[0].logprobs.top_logprobs[0]
        sorted_tokens = sorted(top_tokens.items(), key=lambda x: -x[1])[:10]

        # Show output
        best_token = max(top_tokens, key=top_tokens.get)
        completed_text = prompt + best_token
        st.markdown("### ✅ Completed Text")
        st.success(completed_text)

        st.markdown("### 🔢 Top 10 Predicted Tokens")
        for token, logprob in sorted_tokens:
            prob = round(100 * (10 ** logprob), 2)
            st.write(f"`{token}` — **{prob}%**")

    except Exception as e:
        st.error(f"Error: {e}")
