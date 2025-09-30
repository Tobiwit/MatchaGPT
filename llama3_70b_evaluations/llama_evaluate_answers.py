import csv
import subprocess
import re
import sys
from typing import Optional

def query_ollama_model(model: str, prompt: str, answer: str, timeout: int = 2147483647) -> Optional[int]:
    system_prompt = f"""Consider the following prompt: {prompt}. And consider the following answer: {answer}. Give a rating from 1 to 10 how well the answer answers the prompt. Only answer with the score."""
    
    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=system_prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout
        )
        
        if result.returncode != 0:
            print(f"Error running ollama: {result.stderr.decode('utf-8')}")
            return None
        
        output = result.stdout.decode('utf-8').strip()
        
        rating = extract_rating(output)
        return rating
        
    except subprocess.TimeoutExpired:
        print("Timeout occurred while querying model")
        return None
    except Exception as e:
        print(f"Error querying model: {e}")
        return None

def extract_rating(text: str) -> Optional[int]:
    patterns = [
        r'\b([1-9]|10)\b',
        r'([1-9]|10)/10',
        r'rating:?\s*([1-9]|10)',
        r'score:?\s*([1-9]|10)',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            try:
                rating = int(matches[0])
                if 1 <= rating <= 10:
                    return rating
            except ValueError:
                continue
    
    return None

def process_csv(input_file: str, output_file: str, model: str = "llama3:70b"):
    try:
        with open(input_file, 'r', encoding='windows-1252') as infile:
            reader = csv.DictReader(infile, delimiter=';')
            fieldnames = reader.fieldnames
            
            if not fieldnames:
                print("Error: CSV file appears to be empty or invalid")
                return

            required_columns = ['sess_id', 'prompt', 'answer', 'energy_consumed_kWh']
            missing_columns = [col for col in required_columns if col not in fieldnames]
            if missing_columns:
                print(f"Error: Missing required columns: {missing_columns}")
                return
            
            output_fieldnames = list(fieldnames) + ['rating']
            
            rows_data = []
            
            print(f"Processing {input_file}...")
            
            for i, row in enumerate(reader, 1):
                sess_id = row['sess_id']
                prompt = row['prompt']
                answer = row['answer']
                kwh = row['energy_consumed_kWh']
                
                print(f"Processing row {i} (sess_id: {sess_id})...", end=' ')
                
                rating = query_ollama_model(model, prompt, answer)
                
                if rating is not None:
                    print(f"Rating: {rating}")
                else:
                    print("Rating: Failed to extract")
                
                row_with_rating = row.copy()
                row_with_rating['rating'] = rating if rating is not None else 'N/A'
                rows_data.append(row_with_rating)
                
        
        with open(output_file, 'w', newline='', encoding='windows-1252') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=output_fieldnames, delimiter=';')
            writer.writeheader()
            writer.writerows(rows_data)
        
        print(f"\nCompleted! Results saved to {output_file}")
        
        successful_ratings = sum(1 for row in rows_data if row['rating'] != 'N/A')
        total_rows = len(rows_data)
        print(f"Successfully rated {successful_ratings}/{total_rows} entries")
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found")
    except Exception as e:
        print(f"Error processing CSV: {e}")

def main():
    if len(sys.argv) < 3:
        print("Usage: python evaluate_answers.py <input_csv> <output_csv>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    model = "llama3:70b"
    
    print(f"Using model: {model}")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print()
    
    process_csv(input_file, output_file, model)

if __name__ == "__main__":
    main()