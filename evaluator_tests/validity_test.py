import pandas as pd
import numpy as np
import time
import ollama

EVAL_TIMES = 1  # Optional: Mehrmals ausführen für Mittelwertbildung
MODEL_NAME = "gemma3:27b"

def build_validity_prompt(question, answer, explanation):
    return (
        f"Consider the following prompt: {question}\n"
        f"And the following answer: {answer}\n{explanation}"
        f"Rate on a scale from 1 to 10 how well the answer answers the prompt."
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

def run_validity_test(input_csv_path, output_excel_path):
    df = pd.read_csv(input_csv_path)
    results = []

    for index, row in df.iterrows():

        question = str(row['Question'])
        explanation = str(row['Explanation'])

        for i in range(1):  # 1 richtige + 3 falsche Antworten
            col_name = 'Correct Answer' if i == 0 else f'Incorrect Answer {i}'
            answer = str(row[col_name])

            print(f"🧪 Bewertet {col_name} zu Frage {index + 1}/{len(df)}…")
            scores = []

            for _ in range(EVAL_TIMES):
                eval_prompt = build_validity_prompt(question, answer, explanation)
                score = evaluate_with_ollama_chat(eval_prompt)
                if score is not None:
                    scores.append(score)
                else:
                    print("⚠️ Keine Bewertung erhalten")
                time.sleep(0.2)

            mean_score = np.mean(scores) if scores else None
            var_score = np.var(scores) if scores else None

            results.append({
                'question': question,
                'answer_type': col_name,
                'answer': answer,
                'mean_score': mean_score,
                'variance': var_score,
                'all_scores': scores
            })

    output_df = pd.DataFrame(results)
    output_df.to_excel(output_excel_path, index=False)
    print(f"\n✅ Validitätsergebnisse gespeichert unter: {output_excel_path}")

# Beispielaufruf
if __name__ == "__main__":
    input_file = "ValidityTest.csv"
    output_file = "validitaet_output.xlsx"
    run_validity_test(input_file, output_file)
