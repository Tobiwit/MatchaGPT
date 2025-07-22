import requests
from codecarbon import EmissionsTracker

tracker = EmissionsTracker(project_name="Modell_Picker")

tracker.start()

OLLAMA_URL = "http://localhost:11434/api/generate"

# Modellzuordnung je nach Komplexität
MODEL_MAP = {
    "niedrig": "gemma2:2b",
    "mittel": "mistral:7b",
    "hoch": "deepseek:8b"
}

def ask_model(prompt, model):
    response = requests.post(OLLAMA_URL, json={
        "model": model,
        "prompt": prompt,
        "stream": False
    })
    data = response.json()
    return data.get("response", "").strip()

def classify_complexity(user_prompt):
    classification_prompt = (
        "Stufe bitte diesen Prompt nach Komplexität ein: "
        f"{user_prompt}\n\n"
        "Es ist wichtig, dass du mir als Anwtwort nur ein einziges Wort zurückgibst. Entweder 'niedrig', 'mittel' oder 'hoch'."
    )
    
    result = ask_model(classification_prompt, "gemma2:2b").lower()
    print(f"[DEBUG] Klassifizierungs-Antwort: '{result}'")

    if result not in MODEL_MAP:
        print("⚠️ Komplexität unklar, nehme 'mittel'. Antwort war:", result)
        return "mittel"
    return result

def main():
    user_prompt = input("> Gib deinen Prompt ein:\n> ")
    complexity = classify_complexity(user_prompt)
    selected_model = MODEL_MAP[complexity]
    print(f"\n🧠 Komplexität erkannt als: **{complexity.upper()}** → nutze Modell: {selected_model}")

    answer = ask_model(user_prompt, selected_model)
    print("\n🗨️ Antwort des Modells:")
    print(answer)

if __name__ == "__main__":
    main()

tracker.stop()