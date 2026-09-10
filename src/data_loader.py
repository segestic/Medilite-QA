import pandas as pd
from datasets import load_dataset, concatenate_datasets

def get_prepared_datasets(tokenizer):
    """
    Loads exclusively train splits (to prevent benchmark leakage) and applies
    the exact ChatML format logic from the original notebook.
    """
    print("Loading datasets (Train splits only)...")
    medmcqa_dataset = load_dataset("openlifescienceai/medmcqa", split="train")
    pubmedqa_dataset = load_dataset("bigbio/pubmed_qa", trust_remote_code=True, split="train")
    medqa_dataset = load_dataset("GBaker/MedQA-USMLE-4-options", split="train")

    # ==========================================
    # 1. Exact Original Extraction Functions
    # ==========================================
    
    # --- MedMCQA ---
    def doc_to_text_medmcqa(doc):
        choices = [doc.get("opa", ""), doc.get("opb", ""), doc.get("opc", ""), doc.get("opd", "")]
        option_choices = {"A": choices[0], "B": choices[1], "C": choices[2], "D": choices[3]}
        prompt = f"{doc.get('question', '')}\n"
        choice_ = "\n".join([f"{option}. {text}" for option, text in option_choices.items()])
        return prompt, choice_

    def doc_to_answer_medmcqa(doc):
        correct_answer = ["A", "B", "C", "D"][doc.get('cop', -1)] if 0 <= doc.get('cop', -1) <= 3 else "NULL"
        return correct_answer  

    # --- PubMedQA ---
    def doc_to_text_pubmedqa(doc):
        ctxs = "\n".join(doc.get("CONTEXTS", []))
        prompt = f"Abstract: {ctxs}\nQuestion: {doc.get('QUESTION', '')}\n"
        choice_ = "A. Yes\nB. No\nC. Maybe\n"
        return prompt, choice_

    def doc_to_answer_pubmedqa(doc):
        # Maps "yes", "no", "maybe" to A, B, C for consistency with other datasets
        decision = doc.get('final_decision', '').lower()
        if decision == 'yes': return 'A'
        if decision == 'no': return 'B'
        if decision == 'maybe': return 'C'
        return 'NULL'

    # --- MedQA (USMLE) ---
    def doc_to_text_medqa(doc):
        prompt = f"{doc.get('question', '')}\n"
        options = doc.get('options', {})
        choice_ = "\n".join([f"{key}. {val}" for key, val in options.items()])
        return prompt, choice_

    def doc_to_answer_medqa(doc):
        return doc.get('answer_idx', 'NULL')

    # ==========================================
    # 2. Process and map into standard columns
    # ==========================================
    print("Mapping to Instruction/Input/Output format...")
    
    medmcqa_sliced = medmcqa_dataset.map(
        lambda x: {'instruction': doc_to_text_medmcqa(x)[0], 'input': doc_to_text_medmcqa(x)[1], 'output': doc_to_answer_medmcqa(x)},
        remove_columns=medmcqa_dataset.column_names
    )

    pubmedqa_sliced = pubmedqa_dataset.map(
        lambda x: {'instruction': doc_to_text_pubmedqa(x)[0], 'input': doc_to_text_pubmedqa(x)[1], 'output': doc_to_answer_pubmedqa(x)},
        remove_columns=pubmedqa_dataset.column_names
    )
    
    
    medqa_sliced = medqa_dataset.map(
        lambda x: {'instruction': doc_to_text_medqa(x)[0], 'input': doc_to_text_medqa(x)[1], 'output': doc_to_answer_medqa(x)},
        remove_columns=medqa_dataset.column_names
    )

    combined_dataset = concatenate_datasets([medmcqa_sliced, pubmedqa_sliced, medqa_sliced])

    # ==========================================
    # 3. Apply exact ChatML formatting
    # ==========================================
    print("Applying ChatML formatting...")
    def create_message_column(row):
        messages = []
        user = {
            "content": f"{row['instruction']}\n Input: {row['input']}",
            "role": "user"
        }
        messages.append(user)
        assistant = {
            "content": f"{row['output']}",
            "role": "assistant"
        }
        messages.append(assistant)
        return {"messages": messages}

    def format_dataset_chatml(row):
        return {"text": tokenizer.apply_chat_template(row["messages"], add_generation_prompt=False, tokenize=False)}

    dataset_chatml = combined_dataset.map(create_message_column)
    dataset_chatml = dataset_chatml.map(format_dataset_chatml)
    
    # Create an eval split since TrainingArguments demands `do_eval=True`
    print("Splitting into Train/Eval...")
    dataset_split = dataset_chatml.train_test_split(test_size=0.05, seed=1234)
    
    return dataset_split["train"], dataset_split["test"]
