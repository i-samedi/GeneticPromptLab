# --- START OF FILE example.py ---

import pandas as pd
import json
from GeneticPromptLab.qa_optim import QuestionsAnswersOptimizer
from GeneticPromptLab.ollama_ssh_client import OllamaSSHClient 
import traceback

def trec():
    train_path = './data/trec_train.csv'
    test_path = './data/trec_test.csv'
    model_name = 'multi-qa-MiniLM-L6-cos-v1'
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    with open("./data/trec_label_dict.json", "r") as f:
        label_map = json.load(f)
        label_dict = {str(i):v for i,v in enumerate(label_map)}
    problem_description = "Data are collected from four sources: 4,500 English questions. Your objective is to classify these into one of the following labels: "+str(list(label_dict.values()))

    train_questions_list, train_answers_label, test_questions_list, test_answers_label = train_data['question'].tolist(), train_data['label'].tolist(), test_data['question'].tolist(), test_data['label'].tolist()
    return problem_description, train_questions_list, train_answers_label, test_questions_list, test_answers_label, label_dict, model_name

def agnews():
    train_path = './data/ag_news_train.csv'
    test_path = './data/ag_news_test.csv'
    model_name = 'multi-qa-MiniLM-L6-cos-v1'
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    with open("./data/ag_news_label_dict.json", "r") as f:
        label_map = json.load(f)
        label_dict = {str(i):v for i,v in enumerate(label_map)}
    problem_description = "AG is a collection of more than 1 million news articles. News articles have been gathered from more than 2000 news sources. Your objective is a classification label, with possible values including: "+str(list(label_dict.values()))

    train_questions_list, train_answers_label, test_questions_list, test_answers_label = train_data['question'].tolist(), train_data['label'].tolist(), test_data['question'].tolist(), test_data['label'].tolist()
    return problem_description, train_questions_list, train_answers_label, test_questions_list, test_answers_label, label_dict, model_name

def run_experiment(client, name, data_loader, population_size, generations, sample_p, num_retries):
    """Función auxiliar para ejecutar un experimento completo."""
    print(f"\n--- INICIANDO EXPERIMENTO: {name.upper()} ---")
    problem_description, train_questions_list, train_answers_label, test_questions_list, test_answers_label, label_dict, model_name = data_loader()
    
    lab = QuestionsAnswersOptimizer(
        client=client,
        problem_description=problem_description,
        train_questions_list=train_questions_list,
        train_answers_label=train_answers_label,
        test_questions_list=test_questions_list,
        test_answers_label=test_answers_label,
        label_dict=label_dict,
        model_name=model_name,
        sample_p=sample_p,
        init_and_fitness_sample=population_size,
        window_size_init=2,
        generations=generations,
        num_retries=num_retries
    )
    
    optimized_prompts = lab.genetic_algorithm()
    print("\n--- PROMPTS OPTIMIZADOS FINALES ---")
    print(optimized_prompts)
    print(f"--- EXPERIMENTO {name.upper()} COMPLETADO ---\n")

def main():
    # El cliente se crea UNA SOLA VEZ aquí.
    client = None
    try:
        print("Creando cliente Ollama SSH. Esto puede tardar un momento...")
        client = OllamaSSHClient()
        print("Cliente Ollama SSH creado con éxito.")
        
        # Configuración común para los experimentos
        population_size = 8
        generations = 10 # Puedes bajarlo a 2 o 3 para pruebas rápidas
        sample_p = 0.01
        num_retries = 1

        # Ejecutar experimento AGNEWS
        run_experiment(client, "agnews", agnews, population_size, generations, sample_p, num_retries)

        # Ejecutar experimento TREC
        run_experiment(client, "trec", trec, population_size, generations, sample_p, num_retries)

    except Exception as e:
        print(f"\nOcurrió un error fatal en la ejecución: {e}")
        traceback.print_exc()
    finally:
        # Asegurarnos de cerrar la conexión SSH al finalizar, sin importar si hubo errores.
        if client:
            client.close()

if __name__=='__main__':
    main()

