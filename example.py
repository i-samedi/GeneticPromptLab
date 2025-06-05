#from openai import OpenAI
import pandas as pd
import json
import os # Add
from GeneticPromptLab import QuestionsAnswersOptimizer 

# with open("openai_api.key", "r") as f:
#     key = f.read()
# client = OpenAI(api_key=key.strip())

def load_label_dict(json_path):
    with open(json_path, "r") as f:
        label_data = json.load(f)
    # Ensure integer keys for label_dict, as qa_optim.py uses self.label_dict[int(answer_label_id)]
    if isinstance(label_data, list): # e.g. ["World", "Sports"]
        return {i: v for i, v in enumerate(label_data)}
    elif isinstance(label_data, dict): # e.g. {"0": "World", "1": "Sports"}
        return {int(k): v for k, v in label_data.items()}
    raise ValueError(f"Unsupported label_dict format in {json_path}")

def trec():
    # Configuration
    train_path = './data/trec_train.csv'
    test_path = './data/trec_test.csv'
    label_dict_path = "./data/trec_label_dict.json"
    model_name = 'multi-qa-MiniLM-L6-cos-v1'
    
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    label_dict = load_label_dict(label_dict_path)
    
    # with open("./data/trec_label_dict.json", "r") as f:
    #     label_dict = json.load(f)
    #     label_dict = {i:v for i,v in enumerate(label_dict)}
        
    problem_description = "TREC Question Classification: Classify questions into categories like Abbreviation, Entity, Description, Human, Location, Numeric. Labels are: "+str(list(label_dict.values()))

    # train_questions_list, train_answers_label, test_questions_list, test_answers_label = train_data['question'].tolist(), train_data['label'].tolist(), test_data['question'].tolist(), test_data['label'].tolist()
    # Create GeneticPromptLab instance
    train_questions_list = train_data['question'].tolist()
    train_answers_label = [int(lbl) for lbl in train_data['label'].tolist()] 
    test_questions_list = test_data['question'].tolist()
    test_answers_label = [int(lbl) for lbl in test_data['label'].tolist()]
    
    return problem_description, train_questions_list, train_answers_label, test_questions_list, test_answers_label, label_dict, model_name
    
def agnews():
    train_path = './data/ag_news_train.csv'
    test_path = './data/ag_news_test.csv'
    label_dict_path = "./data/ag_news_label_dict.json"
    model_name = 'multi-qa-MiniLM-L6-cos-v1' # For sentence embeddings

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    label_dict = load_label_dict(label_dict_path)

    problem_description = "AG News Classification: Classify news articles into World, Sports, Business, Sci/Tech. Labels are: "+str(list(label_dict.values()))

    train_questions_list = train_data['question'].tolist()
    train_answers_label = [int(lbl) for lbl in train_data['label'].tolist()]
    test_questions_list = test_data['question'].tolist()
    test_answers_label = [int(lbl) for lbl in test_data['label'].tolist()]

    return problem_description, train_questions_list, train_answers_label, test_questions_list, test_answers_label, label_dict, model_name

def main():
    print("AGNEWS (Ollama Backend):")
    problem_desc_ag, train_q_ag, train_a_ag, test_q_ag, test_a_ag, labels_ag, model_sbert_ag = agnews()

    # GA Parameters for Ollama (likely need to be smaller/slower than OpenAI)
    pop_size = 4        # Smaller population for faster testing
    num_generations = 2 # Fewer generations
    sample_for_init_fitness = 5 # Fewer samples for fitness eval / init prompt gen
    llm_retries = 0     # Retries for each LLM call (0 means 1 attempt)
    llm_pause = 10.0    # Longer pause for Ollama

    lab_agnews = QuestionsAnswersOptimizer(
        client=None, # OpenAI client not used for Ollama
        problem_description=problem_desc_ag, 
        train_questions_list=train_q_ag, 
        train_answers_label=train_a_ag, 
        test_questions_list=test_q_ag, # For completeness, not used in GA loop
        test_answers_label=test_a_ag,   # For completeness
        label_dict=labels_ag, 
        model_name=model_sbert_ag,      # SBERT model for embeddings
        sample_p=0.1,                   # Use 10% of training data for embeddings/sampling pool
        init_and_fitness_sample=sample_for_init_fitness, 
        window_size_init=1,             # Number of examples per initial prompt generation context
        generations=num_generations,
        num_retries=llm_retries,
        population_size_ga=pop_size,    # Pass population size to GA
        pause_value=llm_pause           # Pass pause value
    )
    
    print(f"Starting Genetic Algorithm for AGNEWS with Ollama. This will be slow. Population: {pop_size}, Gens: {num_generations}")
    optimized_prompts_ag = lab_agnews.genetic_algorithm(
        mutation_rate=0.2, 
        elite_size_fraction=0.25, # e.g. 1 elite for pop_size=4
        random_injection_fraction=0.25 # e.g. 1 random for pop_size=4
    )
    print("\nAGNEWS - Final population prompts from Ollama GA:")
    if optimized_prompts_ag:
        for i, p_text in enumerate(optimized_prompts_ag):
            print(f"  Prompt {i+1}: {p_text[:100]}...")
    else:
        print("  No prompts returned from AGNEWS GA.")
    print("-------- AGNEWS EXPERIMENT COMPLETED (Ollama) --------\n")
    
    
    print("TREC (Ollama Backend):")
    problem_desc_tr, train_q_tr, train_a_tr, test_q_tr, test_a_tr, labels_tr, model_sbert_tr = trec()

    lab_trec = QuestionsAnswersOptimizer(
        client=None,
        problem_description=problem_desc_tr, 
        train_questions_list=train_q_tr, 
        train_answers_label=train_a_tr, 
        test_questions_list=test_q_tr,
        test_answers_label=test_a_tr,
        label_dict=labels_tr, 
        model_name=model_sbert_tr,
        sample_p=0.1, # Using a sample of TREC data
        init_and_fitness_sample=sample_for_init_fitness,
        window_size_init=1,
        generations=num_generations,
        num_retries=llm_retries,
        population_size_ga=pop_size,
        pause_value=llm_pause
    )
    print(f"Starting Genetic Algorithm for TREC with Ollama. Population: {pop_size}, Gens: {num_generations}")
    optimized_prompts_tr = lab_trec.genetic_algorithm(
        mutation_rate=0.2,
        elite_size_fraction=0.25,
        random_injection_fraction=0.25
    )
    print("\nTREC - Final population prompts from Ollama GA:")
    if optimized_prompts_tr:
        for i, p_text in enumerate(optimized_prompts_tr):
            print(f"  Prompt {i+1}: {p_text[:100]}...")
    else:
        print("  No prompts returned from TREC GA.")
    print("-------- TREC EXPERIMENT COMPLETED (Ollama) --------")


if __name__=='__main__':
    # Ensure data files are present by checking one of them
    if not os.path.exists('./data/ag_news_train.csv'):
        print("Data files not found. Please run `python example_data_setup.py` first.")
        print("Then, ensure your `ssh_config.json` is in the project root and correctly configured.")
    else:
        main()