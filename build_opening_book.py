import re
import json
import os

def parse_and_save_ffo(input_txt_path, output_json_path):
    opening_book = []
    
    if not os.path.exists(input_txt_path):
        print(f"[!] Error: Could not find '{input_txt_path}'. Please create it and paste your FFO text inside.")
        return

    with open(input_txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        original_line = line.strip()
        if not original_line:
            continue
            
        line_lower = original_line.lower()
        
        # 2. Extract Moves: Find all 2-character coordinates (e.g., c4, e3)
        moves = re.findall(r'[a-h][1-8]', line_lower)
        if not moves:
            continue
            
        # 3. Translate Coordinates (a-h, 1-8 -> 0-63)
        indices = []
        for move in moves:
            col = ord(move[0]) - 97       # 'a' becomes 0, 'h' becomes 7
            row = int(move[1]) - 1        # '1' becomes 0, '8' becomes 7
            idx = row * 8 + col
            indices.append(idx)
            
        # 4. Extract Metadata
        # Get the name by grabbing everything before the first 'c4'
        name_parts = original_line.split('c4')
        name = name_parts[0].strip() if len(name_parts) > 0 and name_parts[0].strip() else "Variante FFO"
        
        # Get the evaluation tag (usually the last word/symbol on the line)
        eval_tag = original_line.split(' ')[-1]

        opening_book.append({
            "name": name,
            "sequence": indices,
            "evaluation": eval_tag
        })

    # 5. Save to JSON
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(opening_book, f, indent=4, ensure_ascii=False)
        
    print(f"Successfully parsed and converted {len(opening_book)} openings!")
    print(f"Saved directly to: {output_json_path}")

if __name__ == "__main__":
    INPUT_FILE = "ffo_openings.txt"
    OUTPUT_FILE = "ffo_openings.json"
    
    parse_and_save_ffo(INPUT_FILE, OUTPUT_FILE)