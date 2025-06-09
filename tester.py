# --- START OF FILE tester.py ---

from GeneticPromptLab.function_templates import function_templates
from GeneticPromptLab.ollama_ssh_client import OllamaSSHClient
from GeneticPromptLab.utils import send_query2gpt         
import pandas as pd
import numpy as np
import json
import os
import time
from tqdm import trange

# La función send_query2gpt que estaba aquí ha sido eliminada para evitar duplicación.

def read_data(path_test_df, path_label_dict):
    # Esta función no necesita cambios
    with open(path_label_dict, "r") as f:
        # Asegurar que las claves del diccionario sean strings para consistencia con JSON
        label_map = json.load(f)
        label_dict = {str(k): v for k, v in label_map.items()}
    df = pd.read_csv(path_test_df)
    questions = df['question'].tolist()
    # Mapear las etiquetas numéricas del CSV a las etiquetas de texto del diccionario
    answers = [label_dict[str(v)] for v in df['label'].tolist()]
    return questions, answers, list(label_dict.values()) # Devuelve vocabulario como lista

def read_latest_epoch_data(run_id):
    # Esta función no necesita cambios
    dir_path = f"./runs/{run_id}/"
    if not os.path.exists(dir_path):
        print(f"Error: El directorio para run_id '{run_id}' no existe en {os.path.abspath('./runs/')}")
        return None
    files = [f for f in os.listdir(dir_path) if f.startswith('epoch_') and f.endswith('.csv')]
    if not files:
        print(f"Error: No se encontraron archivos 'epoch_*.csv' en {dir_path}")
        return None
    # Sort files based on numerical value of epoch_id
    sorted_files = sorted(files, key=lambda x: int(x.split('_')[1].split('.')[0]))
    latest_file_path = os.path.join(dir_path, sorted_files[-1])
    df = pd.read_csv(latest_file_path)
    return df

def get_highest_fitness_prompt(df):
    # Esta función no necesita cambios
    if df is None or df.empty:
        return None
    max_fitness_row = df.loc[df['Fitness Score'].idxmax()]
    highest_fitness_prompt = max_fitness_row['Prompt']
    return highest_fitness_prompt

def ag_news(client): # <-- La función ahora recibe el cliente como argumento
    run_id = "XrFnn68pnF"
    path_test_df = "./data/ag_news_test.csv"
    path_label_dict = "./data/ag_news_label_dict.json"
    
    questions, answers, label_vocab = read_data(path_test_df=path_test_df, path_label_dict=path_label_dict)
    
    latest_epoch_df = read_latest_epoch_data(run_id)
    best_prompt = get_highest_fitness_prompt(latest_epoch_df)
    
    if not best_prompt:
        print("No se pudo obtener el mejor prompt. Saltando test de AG News.")
        return

    batch_size = 10
    qa_function_template = function_templates[1].copy() # Usar una copia para no modificar el original
    qa_function_template["parameters"]["properties"]["label_array"]["items"]["properties"]["label"]["enum"] = label_vocab
    qa_function_template["parameters"]["properties"]["label_array"]["items"]["properties"]["label"]["description"] += str(label_vocab)
    qa_function_template["parameters"]["properties"]["label_array"]["minItems"] = batch_size
    qa_function_template["parameters"]["properties"]["label_array"]["maxItems"] = batch_size
    
    aggregate_accuracy = []
    batches_skipped_count = 0
    for i in trange(0, len(questions), batch_size, desc="testing_agnews"):
        question_subset = questions[i:i+batch_size]
        answer_subset = answers[i:i+batch_size]
        questions_list = "\n\n".join([str(j+1)+'. """'+question+'"""' for j,question in enumerate(question_subset)])
        
        # Asegurar que el batch final tenga el tamaño correcto si es más pequeño
        current_batch_size = len(question_subset)
        if current_batch_size != batch_size:
            qa_function_template["parameters"]["properties"]["label_array"]["minItems"] = current_batch_size
            qa_function_template["parameters"]["properties"]["label_array"]["maxItems"] = current_batch_size

        try:
            response = [v['label'] for v in send_query2gpt(client, [{"role": "system", "content": best_prompt}, {"role": "user", "content": "Questions:\n"+questions_list}], qa_function_template, temperature=0.0, pause=5)['label_array']]
            accuracy = sum(1 if a == b else 0 for a, b in zip(response, answer_subset)) / len(response)
            aggregate_accuracy.append(accuracy)
        except Exception as e:
            print(f"Error en el batch {i//batch_size}, saltando. Error: {e}")
            batches_skipped_count += 1
            
    print("Batches skipped:", batches_skipped_count)
    if aggregate_accuracy:
        print("Accuracy:", str(round(100*np.mean(aggregate_accuracy), 3))+"%")
    else:
        print("No se pudo calcular la precisión, no se completó ningún batch.")

def trec(client): # <-- La función ahora recibe el cliente como argumento
    run_id = "08zLX4cd97"
    path_test_df = "./data/trec_test.csv"
    path_label_dict = "./data/trec_label_dict.json"

    questions, answers, label_vocab = read_data(path_test_df=path_test_df, path_label_dict=path_label_dict)

    latest_epoch_df = read_latest_epoch_data(run_id)
    best_prompt = get_highest_fitness_prompt(latest_epoch_df)

    if not best_prompt:
        print("No se pudo obtener el mejor prompt. Saltando test de TREC.")
        return

    batch_size = 10
    qa_function_template = function_templates[1].copy() # Usar una copia
    qa_function_template["parameters"]["properties"]["label_array"]["items"]["properties"]["label"]["enum"] = label_vocab
    qa_function_template["parameters"]["properties"]["label_array"]["items"]["properties"]["label"]["description"] += str(label_vocab)
    qa_function_template["parameters"]["properties"]["label_array"]["minItems"] = batch_size
    qa_function_template["parameters"]["properties"]["label_array"]["maxItems"] = batch_size
    
    aggregate_accuracy = []
    batches_skipped_count = 0
    for i in trange(0, len(questions), batch_size, desc="testing_trec"):
        question_subset = questions[i:i+batch_size]
        answer_subset = answers[i:i+batch_size]
        questions_list = "\n\n".join([str(j+1)+'. """'+question+'"""' for j,question in enumerate(question_subset)])

        current_batch_size = len(question_subset)
        if current_batch_size != batch_size:
            qa_function_template["parameters"]["properties"]["label_array"]["minItems"] = current_batch_size
            qa_function_template["parameters"]["properties"]["label_array"]["maxItems"] = current_batch_size

        try:
            response = [v['label'] for v in send_query2gpt(client, [{"role": "system", "content": best_prompt}, {"role": "user", "content": "Questions:\n"+questions_list}], qa_function_template, temperature=0.0, pause=5)['label_array']]
            accuracy = sum(1 if a == b else 0 for a, b in zip(response, answer_subset)) / len(response)
            aggregate_accuracy.append(accuracy)
        except Exception as e:
            print(f"Error en el batch {i//batch_size}, saltando. Error: {e}")
            batches_skipped_count += 1
            
    print("Batches skipped:", batches_skipped_count)
    if aggregate_accuracy:
        print("Accuracy:", str(round(100*np.mean(aggregate_accuracy), 3))+"%")
    else:
        print("No se pudo calcular la precisión.")

def main():
    client = None
    try:
        # 3. Crear una única instancia del cliente Ollama al inicio.
        print("Creando cliente Ollama SSH para el tester...")
        client = OllamaSSHClient()
        print("Cliente Ollama SSH creado con éxito.")

        # Llamar a las funciones de test, pasando el cliente como argumento.
        ag_news(client)
        # trec(client)

    except Exception as e:
        print(f"Ocurrió un error fatal en el script principal del tester: {e}")
    finally:
        # 4. Asegurarse de cerrar la conexión al final.
        if client:
            client.close()

if __name__=='__main__':
    main()

# --- END OF FILE tester.py ---