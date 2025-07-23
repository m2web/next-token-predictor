import os
import openai
from openai import OpenAI

# Initialize client using new SDK (make sure your key is set as an env var)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_next_token_probs(prompt, top_k=10, model="gpt-3.5-turbo-instruct"):
    try:
        response = client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=1,
            logprobs=top_k,
            temperature=0.7,
        )
        top_tokens = response.choices[0].logprobs.top_logprobs[0]
        return top_tokens
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def chat_interface():
    print("🔮 ChatGPT Token Predictor Interface (OpenAI SDK v1+)")
    print("Type 'exit' to quit.\n")
    prompt_history = ""
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in {"exit", "quit"}:
            break
        prompt_history += user_input + "\n"

        top_predictions = get_next_token_probs(prompt_history)
        if top_predictions:
            print("\n📊 Top Predicted Next Tokens:")
            for token, logprob in top_predictions.items():
                prob = round(100 * (10 ** logprob), 5)
                print(f"  '{token}': {prob}%")
            print()
        else:
            print("No predictions available.\n")

if __name__ == "__main__":
    chat_interface()
