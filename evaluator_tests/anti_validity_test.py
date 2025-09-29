import pandas as pd
import numpy as np
import time
import ollama

EVAL_TIMES = 1
MODEL_NAME = "llama3:70b"  # oder 'llama3', 'phi3', etc.

def build_eval_prompt(prompt, answer):
    return (
        f"Consider the following prompt: {prompt}\n"
        f"And the following answer: {answer}\n"
        f"Rate on a scale from 1 to 10 how well the answer answers the prompt. "
        f"Only return a single number from 1 to 10."
    )

def evaluate_with_ollama_chat(prompt_text):
    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are an impartial evaluator. Only respond with a number from 1 to 10."},
                {"role": "user", "content": prompt_text}
            ]
        )
        content = response['message']['content'].strip()
        digits = ''.join(filter(str.isdigit, content))
        if digits:
            score = int(digits)
            return min(max(score, 1), 10)
    except Exception as e:
        print(f"[Fehler] {e}")
    return None

def run_consistency_test(input_excel_path, output_excel_path):
    df = pd.read_excel(input_excel_path)
    results = []

    for index, row in df.iterrows():
        prompt = str(row['Question'])
        answer = str(row['Answer'])

        print(f"⭐ Verarbeite Eintrag {index + 1}/{len(df)}…")
        scores = []

        for i in range(EVAL_TIMES):
            eval_prompt = build_eval_prompt(prompt, answer)
            score = evaluate_with_ollama_chat(eval_prompt)
            if score is not None:
                scores.append(score)
            else:
                print(f"⚠️ Keine Bewertung bei Versuch {i+1}")
            time.sleep(0.2)  # kleine Pause gegen Überlastung

        mean_score = np.mean(scores) if scores else None
        var_score = np.var(scores) if scores else None

        results.append({
            'prompt': prompt,
            'answer': answer,
            'mean_score': mean_score,
            'variance': var_score,
            'all_scores': scores
        })

    output_df = pd.DataFrame(results)
    output_df.to_excel(output_excel_path, index=False)
    print(f"\n✅ Ergebnisse gespeichert unter: {output_excel_path}")

# Beispielaufruf
if __name__ == "__main__":
    input_file = "anti_validity_questions.xlsx"
    output_file = "bewertung_output.xlsx"
    run_consistency_test(input_file, output_file)
