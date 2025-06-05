import random
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import string
from tqdm import tqdm
from .utils import send_query2gpt 
from .function_templates import function_templates
import warnings
from .base_class import GeneticPromptLab 
import copy # NEW ADD

warnings.filterwarnings("ignore", message="`resume_download` is deprecated and will be removed in version 1.0.0.")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
class QuestionsAnswersOptimizer(GeneticPromptLab):
    # def __init__(self, client, problem_description, train_questions_list, train_answers_label, test_questions_list, test_answers_label, label_dict, model_name, sample_p=1.0, init_and_fitness_sample=10, window_size_init=1, generations=10, num_retries=1):
    #     self.num_retries = num_retries
    #     self.client = client
    #     self.generations = generations
    #     self.init_and_fitness_sample = init_and_fitness_sample
    #     self.test_questions_list = test_questions_list
    #     self.test_answers_label = test_answers_label
    #     self.label_dict = label_dict
    #     self.problem_description = problem_description
    #     self.window_size_init = window_size_init

    #     self.model = SentenceTransformer(model_name)
    #     self.sample_p = sample_p
    #     train_indices_list = np.random.choice(np.arange(len(train_questions_list)), size=int(len(train_questions_list)*self.sample_p))
    #     self.train_questions_list = [train_questions_list[i] for i in train_indices_list]
    #     self.train_answers_label = [train_answers_label[i] for i in train_indices_list]
    #     self.embeddings = self.model.encode(self.train_questions_list, show_progress_bar=True)
    #     self.already_sampled_indices = set()
    
    def __init__(self, client, problem_description, train_questions_list, train_answers_label, 
                 test_questions_list, test_answers_label, label_dict, model_name, 
                 sample_p=1.0, init_and_fitness_sample=10, window_size_init=1, 
                 generations=10, num_retries=1, population_size_ga=8, pause_value=5.0): # Added population_size_ga, pause_value

        self.client = client # Kept for signature, but Ollama version of send_query2gpt won't use it
        self.problem_description = problem_description
        self.label_dict = label_dict # Expects int keys if train_answers_label has ints: {0: "World", 1: "Sports"}
        self.label_vocab_list = list(label_dict.values()) # For Ollama prompt

        self.model = SentenceTransformer(model_name) # For embeddings
        
        # Sampling training data
        self.sample_p = sample_p
        if len(train_questions_list) == 0:
            raise ValueError("train_questions_list cannot be empty.")
        
        num_to_sample = int(len(train_questions_list) * self.sample_p)
        if num_to_sample == 0 and len(train_questions_list) > 0: # Ensure at least one sample if original list not empty
            num_to_sample = 1
        
        if num_to_sample > 0:
            train_indices_list = np.random.choice(np.arange(len(train_questions_list)), size=num_to_sample, replace=False)
            self.train_questions_list = [train_questions_list[i] for i in train_indices_list]
            self.train_answers_label = [train_answers_label[i] for i in train_indices_list]
        else: # If original list was empty or sample_p was 0 leading to 0 samples
            self.train_questions_list = []
            self.train_answers_label = []

        if self.train_questions_list:
            self.embeddings = self.model.encode(self.train_questions_list, show_progress_bar=True)
        else:
            self.embeddings = np.array([]) # Empty embeddings if no training data

        self.already_sampled_indices = set() # For distinct sampling

        # GA Parameters
        self.generations = generations
        self.init_and_fitness_sample = min(init_and_fitness_sample, len(self.train_questions_list)) if self.train_questions_list else 0
        self.window_size_init = window_size_init
        self.num_retries = num_retries # Retries for LLM calls
        self.pause_value = float(pause_value) # Pause between LLM calls
        self.population_size_ga = population_size_ga # Store population size

        # Test data (currently not used in GA loop, but kept for potential future use)
        self.test_questions_list = test_questions_list
        self.test_answers_label = test_answers_label
        
        # Run ID setup
        self.run_id = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
        self.run_path = os.path.join('runs', self.run_id)
        if not os.path.exists(self.run_path):
            os.makedirs(self.run_path)
        print(f"Run ID: {self.run_id} has been created at {self.run_path}")

    # def create_prompts(self, data):
    #     data_doubled = data+data
    #     prompts = []
    #     for i in range(len(data)):
    #         sample = data_doubled[i:i+self.window_size_init]
    #         sample_prompt = "\n".join(["Question: \"\"\""+s["q"]+"\"\"\"\nCorrect Label:\"\"\""+s["a"]+"\"\"\"" for s in sample])
    #         messages = [{"role": "system", "content": "Problem Description: "+self.problem_description+"\n\n"+function_templates[0]["description"]+"\n\nNote: For this task the labels are: "+"\n".join([str(k)+". "+str(v) for k,v in self.label_dict.items()])}, {"role": "user", "content": "Observe the following samples:\n\n"+sample_prompt}]
    #         prompt = send_query2gpt(self.client, messages, function_templates[0])['prompt']
    #         prompts.append(prompt)
    #     return prompts

    # def generate_init_prompts(self, n=None):
    #     if n is None:
    #         n = self.init_and_fitness_sample
    #     distinct_sample_indices = self.sample_distinct(n)
    #     data = []
    #     for sample_index in distinct_sample_indices:
    #         question = self.train_questions_list[int(sample_index)]
    #         answer = self.train_answers_label[int(sample_index)]
    #         data.append({"q": question, "a": self.label_dict[answer]})
    #     prompts = self.create_prompts(data)
    #     return prompts

    # def sample_distinct(self, n):
    #     embeddings = self.embeddings

    #     if len(self.already_sampled_indices) > 0:
    #         mask = np.ones(len(embeddings), dtype=bool)
    #         mask[list(self.already_sampled_indices)] = False
    #         embeddings = embeddings[mask]

    #     kmeans = KMeans(n_clusters=n, random_state=0).fit(embeddings)
    #     closest_indices, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, embeddings)
    #     sampled_indices = set(closest_indices)

    #     while len(sampled_indices) < n:
    #         remaining_indices = set(range(len(embeddings))) - sampled_indices
    #         remaining_embeddings = embeddings[list(remaining_indices)]
    #         kmeans = KMeans(n_clusters=n - len(sampled_indices), random_state=0).fit(remaining_embeddings)
    #         _, closest_indices = pairwise_distances_argmin_min(kmeans.cluster_centers_, remaining_embeddings)
    #         sampled_indices.update(closest_indices)

    #     sampled_indices = list(sampled_indices)[:n]
    #     self.already_sampled_indices.update(sampled_indices)
    #     return sampled_indices

    # def genetic_algorithm(self, mutation_rate=0.1):
    #     output_directory = "runs"
    #     run_id = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
    #     run_path = os.path.join(output_directory, run_id)
    #     if not os.path.exists(run_path):
    #         os.makedirs(run_path)
    #     print(f"Run ID: {run_id} has been created at {run_path}")
    #     initial_prompts = self.generate_init_prompts()
    #     population = initial_prompts
    #     bar = tqdm(range(self.generations))
    #     for gen_id in bar:
    #         print("Complete Population:",population)
    #         fitness_scores, questions_list, correct_answers_list, prompt_answers_list = self.evaluate_fitness(population)
    #         top_prompts, top_prompts_answers_list = self.select_top_prompts(fitness_scores, population, prompt_answers_list)
    #         df = pd.DataFrame({
    #             'Prompt': population,
    #             'Fitness Score': fitness_scores
    #         })
    #         df.to_csv(os.path.join(run_path, f'epoch_{gen_id}.csv'), index=False)
    #         print()
    #         print("Top Population:", top_prompts)
    #         print("\n\n")
    #         new_prompts = self.crossover_using_gpt(top_prompts, questions_list, correct_answers_list, top_prompts_answers_list)
    #         num_random_prompts = int(self.init_and_fitness_sample * 0.25)
    #         random_prompts = self.generate_init_prompts(num_random_prompts)
    #         population = top_prompts + new_prompts + random_prompts
    #         population = self.mutate_prompts(population, mutation_rate)
    #         bar.set_description(str({"epoch": gen_id+1, "acc": round(float(np.mean(fitness_scores))*100, 1)}))
    #     bar.close()

    #     return population

    # def evaluate_fitness(self, prompts):
    #     distinct_sample_indices = self.sample_distinct(self.init_and_fitness_sample)
    #     just_questions_list = [self.train_questions_list[int(index)] for index in distinct_sample_indices]
    #     questions_list = "\n\n".join([str(i+1)+'. """'+self.train_questions_list[int(index)]+'"""' for i,index in enumerate(distinct_sample_indices)])
    #     correct_answers_list = [self.label_dict[self.train_answers_label[int(i)]] for i in distinct_sample_indices]
    #     acc_list = []
    #     prompt_latest_answers_list = []
    #     for prompt in prompts:
    #         acc = []
    #         for retry_id in range(self.num_retries):
    #             messages = [{"role": "system", "content": prompt}, {"role": "user", "content": "Questions:\n\n"+questions_list+"\n\nNote: Ensure you respond with "+str(len(distinct_sample_indices))+" labels."}]
    #             tmp_function_template = function_templates[1]
    #             tmp_function_template["parameters"]["properties"]["label_array"]["items"]["properties"]["label"]["enum"] = [v for _,v in self.label_dict.items()]
    #             tmp_function_template["parameters"]["properties"]["label_array"]["items"]["properties"]["label"]["description"] += str([v for _,v in self.label_dict.items()])
    #             tmp_function_template["parameters"]["properties"]["label_array"]["minItems"] = len(distinct_sample_indices)
    #             tmp_function_template["parameters"]["properties"]["label_array"]["maxItems"] = len(distinct_sample_indices)
    #             labels = send_query2gpt(self.client, messages, tmp_function_template)
    #             labels = [l['label'] for l in labels['label_array']]
    #             accuracy = sum(1 if a == b else 0 for a, b in zip(labels, correct_answers_list)) / len(labels)
    #             acc.append(accuracy)
    #             prompt_latest_answers_list.append(labels)
    #         acc_list.append(sum(acc)/len(acc))
    #     return acc_list, just_questions_list, correct_answers_list, prompt_latest_answers_list

    # def select_top_prompts(self, fitness_scores, population, prompt_answers_list, top_fraction=0.5):
    #     paired_list = list(zip(population, fitness_scores, prompt_answers_list))
    #     sorted_prompts = sorted(paired_list, key=lambda x: x[1], reverse=True)
    #     cutoff = int(len(sorted_prompts) * top_fraction)
    #     return [prompt for prompt, score, answers_list in sorted_prompts[:cutoff]], [answers_list for prompt, score, answers_list in sorted_prompts[:cutoff]]

    # def crossover_using_gpt(self, prompts, questions_list, correct_answers_list, top_prompts_answers_list):
    #     if len(prompts)<2:
    #         raise Exception("Too few to cross-over.")
    #     new_prompts = []
    #     for i in range(0, len(prompts), 2):
    #         if i + 1 < len(prompts):
    #             template = prompts[i]
    #             additive = prompts[i + 1]
    #             answers_from_the_two_parent_prompts = top_prompts_answers_list[i:i+2]
    #             if template.lower().strip()==additive.lower().strip():
    #                 additive = self.gpt_mutate(additive)
    #             new_prompt = self.gpt_mix_and_match(template, additive, questions_list, correct_answers_list, answers_from_the_two_parent_prompts)
    #             new_prompts.append(new_prompt)
    #     return new_prompts
    
    # def gpt_mutate(self, prompt):
    #     tmp_function_template = function_templates[2]
    #     tmp_function_template["parameters"]["properties"]["mutated_prompt"]["description"] += str(round(random.random(), 3))
    #     messages = [{"role": "system", "content": "You are a prompt-mutator as part of an over-all genetic algorithm. Mutate the following prompt while not detracting from the core-task but still rephrasing/mutating the prompt.\n\n"+"Note: For this task the over-arching Problem Description is: "+self.problem_description}, {"role": "user", "content": "Modify the following prompt: \"\"\""+prompt+'"""'}]
    #     mutated_prompt = send_query2gpt(self.client, messages, tmp_function_template, temperature=random.random()/2+0.5)['mutated_prompt']
    #     return mutated_prompt

    # def gpt_mix_and_match(self, template, additive, questions_list, correct_answers_list, answers_from_parent_prompts):
    #     example = "\n\n".join(['Question: """'+q+'"""\nIdeal Answer: """'+a+'"""\nYour template parent\'s answer: """'+p_0+'"""\nYour additive parent\'s answer: """'+p_1 for q,a,p_0,p_1 in zip(questions_list[:5], correct_answers_list[:5], answers_from_parent_prompts[0], answers_from_parent_prompts[1])])
    #     messages = [{"role": "system", "content": "You are a cross-over system as part of an over-all genetic algorithm. You are to ingrain segments of an additive prompt to that of a template/control prompt to create a healthier offspring.\n\n"+"Note: For this task the over-arching Problem Description is: "+self.problem_description+"\n\nExample & History for context:"+example+"\n\nNote: You can use previous mistakes as stepping stones, to quote words/semantics/phrases/keywords/verbs which you think led to the mistake by the AI."}, {"role": "user", "content": "Template Prompt: \"\"\""+template+'"""\n'+'"""Additive Prompt: """'+additive}]
    #     child_prompt = send_query2gpt(self.client, messages, function_templates[3])['child_prompt']
    #     return child_prompt

    # def mutate_prompts(self, prompts, mutation_rate=0.1):
    #     mutated_prompts = []
    #     for prompt in prompts:
    #         if random.random() < mutation_rate:
    #             mutated_prompts.append(self.gpt_mutate(prompt))
    #         else:
    #             mutated_prompts.append(prompt)
    #     return mutated_prompts
    
    
    def create_prompts(self, data): # data is list of {"q": ..., "a": ...}
        # data_doubled = data+data # Original logic, seems to ensure windowing works at end
        prompts = []
        if not data:
            return []

        # print(f"DEBUG (qa_optim): Creating prompts for {len(data)} data points.")
        for i in range(len(data)):
            # sample = data_doubled[i : i + self.window_size_init]
            # Simpler windowing: take a slice, if too short, it's fine.
            current_window_data = data[i : i + self.window_size_init]
            if not current_window_data: continue # Should not happen if data is not empty

            sample_prompt_text = "\n".join([f"Question: \"\"\"{s['q']}\"\"\"\nCorrect Label: \"\"\"{s['a']}\"\"\"" for s in current_window_data])
            
            # Message construction for 'generate_prompts' (function_templates[0])
            # The description of function_templates[0] is the main system instruction.
            # The "Problem Description" and labels are context for that instruction.
            # The "sample_prompt_text" (examples) is the user content.
            system_message_content = (
                f"Problem Description: {self.problem_description}\n\n"
                f"Note: For this task the labels are:\n" + 
                "\n".join([f"{k}. {v}" for k, v in self.label_dict.items()])
            )
            # `function_templates[0]["description"]` will be added by send_query2gpt

            messages = [
                {"role": "system", "content": system_message_content},
                {"role": "user", "content": f"Observe the following samples:\n\n{sample_prompt_text}"}
            ]
            try:
                # function_templates[0] has name "generate_prompts"
                response_data = send_query2gpt(self.client, messages, function_templates[0], 
                                               temperature=0.7, pause=self.pause_value, retries=self.num_retries)
                if response_data and "prompt" in response_data:
                    prompts.append(response_data['prompt'])
                else:
                    print(f"Warning (qa_optim): Failed to generate prompt for item. Response: {response_data}. Falling back.")
                    prompts.append(f"Fallback prompt: Classify based on these labels: {self.label_vocab_list}") 
            except Exception as e:
                print(f"Error in create_prompts with Ollama: {e}. Falling back.")
                prompts.append(f"Fallback prompt due to error: Classify based on these labels: {self.label_vocab_list}")
        return prompts

    def generate_init_prompts(self, n=None):
        if n is None:
            n = self.init_and_fitness_sample
        
        if n == 0 or not self.train_questions_list: # No samples to draw from or n is 0
            print("Warning (qa_optim): Cannot generate initial prompts, n=0 or no training data.")
            return [f"Default prompt: Please classify text into one of {self.label_vocab_list}."] * self.population_size_ga


        distinct_sample_indices = self.sample_distinct(n)
        if not distinct_sample_indices: # sample_distinct might return empty if n is too large for available data
            print(f"Warning (qa_optim): sample_distinct returned no indices for n={n}. Using random samples or default.")
            if len(self.train_questions_list) >= n:
                 distinct_sample_indices = random.sample(range(len(self.train_questions_list)), n)
            elif self.train_questions_list: # take all if less than n
                 distinct_sample_indices = list(range(len(self.train_questions_list)))
            else: # Should be caught by earlier check
                return [f"Default prompt: Please classify text into one of {self.label_vocab_list}."] * self.population_size_ga


        data_for_prompting = []
        for sample_idx in distinct_sample_indices:
            question = self.train_questions_list[int(sample_idx)]
            answer_label_id = self.train_answers_label[int(sample_idx)]
            # Ensure answer_label_id is int if label_dict keys are int
            answer_text = self.label_dict[int(answer_label_id)] 
            data_for_prompting.append({"q": question, "a": answer_text})
        
        prompts = self.create_prompts(data_for_prompting)
        # Ensure we have enough prompts for the population
        if not prompts: # If create_prompts failed entirely
            prompts = [f"Fallback prompt after create_prompts failed. Classify into: {self.label_vocab_list}"]
        
        final_prompts = (prompts * (self.population_size_ga // len(prompts) + 1))[:self.population_size_ga]
        return final_prompts


    def sample_distinct(self, n):
        if self.embeddings.shape[0] == 0 or n == 0: # No embeddings or requesting 0 samples
            return []
        
        current_n = min(n, self.embeddings.shape[0]) # Cannot sample more than available embeddings
        if current_n == 0: return []

        # Simplified sampling: if already_sampled_indices is too full, reset or handle differently.
        # For now, let's just sample from all available embeddings each time for simplicity,
        # as managing remaining embeddings can be complex.
        # A more robust distinct sampling would filter out already_sampled_indices from self.embeddings.
        
        available_indices = list(set(range(self.embeddings.shape[0])) - self.already_sampled_indices)
        if len(available_indices) < current_n:
            # print(f"Warning (qa_optim): Not enough unique samples remaining ({len(available_indices)} for requested {current_n}). Resetting already_sampled_indices.")
            self.already_sampled_indices = set() # Reset if we run out
            available_indices = list(range(self.embeddings.shape[0]))
            if len(available_indices) < current_n: # Still not enough after reset (e.g. total data < n)
                current_n = len(available_indices)
                if current_n == 0: return []
        
        if not available_indices: return []

        embeddings_to_sample_from = self.embeddings[available_indices, :]
        original_indices_map = {new_idx: original_idx for new_idx, original_idx in enumerate(available_indices)}

        try:
            # K-Means might fail if n_clusters > n_samples
            actual_clusters = min(current_n, embeddings_to_sample_from.shape[0])
            if actual_clusters == 0: return []

            kmeans = KMeans(n_clusters=actual_clusters, random_state=0, n_init='auto').fit(embeddings_to_sample_from)
            # pairwise_distances_argmin_min returns indices relative to the input `embeddings_to_sample_from`
            closest_relative_indices, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, embeddings_to_sample_from)
            
            # Map back to original indices in self.train_questions_list
            sampled_original_indices = [original_indices_map[rel_idx] for rel_idx in closest_relative_indices]
            
            self.already_sampled_indices.update(sampled_original_indices)
            return list(set(sampled_original_indices))[:current_n] # Ensure distinct and correct count

        except Exception as e:
            # print(f"Error in KMeans sampling: {e}. Falling back to random sampling.")
            if available_indices:
                fallback_sampled_relative_indices = random.sample(range(len(available_indices)), min(current_n, len(available_indices)))
                sampled_original_indices = [original_indices_map[rel_idx] for rel_idx in fallback_sampled_relative_indices]
                self.already_sampled_indices.update(sampled_original_indices)
                return list(set(sampled_original_indices))[:current_n]
            return []


    def evaluate_fitness(self, prompts): # Returns acc_list, questions_used, correct_answers_text, all_prompt_answers_text
        if self.init_and_fitness_sample == 0 or not self.train_questions_list:
            print("Warning (qa_optim): Cannot evaluate fitness, init_and_fitness_sample is 0 or no training data.")
            # Return structure consistent with normal flow but with 0 fitness
            dummy_answers = [[] for _ in prompts]
            return [0.0] * len(prompts), [], [], dummy_answers

        distinct_sample_indices = self.sample_distinct(self.init_and_fitness_sample)
        if not distinct_sample_indices:
            print("Warning (qa_optim): sample_distinct returned no indices for fitness evaluation.")
            dummy_answers = [[] for _ in prompts]
            return [0.0] * len(prompts), [], [], dummy_answers


        questions_for_eval_text = [self.train_questions_list[int(idx)] for idx in distinct_sample_indices]
        # String format for user prompt to LLM
        questions_list_str = "\n\n".join([f"{i+1}. \"\"\"{self.train_questions_list[int(idx)]}\"\"\"" for i, idx in enumerate(distinct_sample_indices)])
        
        # Correct answers as text
        correct_answers_text = [self.label_dict[int(self.train_answers_label[int(idx)])] for idx in distinct_sample_indices]
        
        current_batch_size = len(distinct_sample_indices)
        all_fitness_scores = []
        all_prompt_answers_text = [] # Store text answers for each prompt

        for prompt_idx, prompt_text in enumerate(prompts):
            accumulated_accuracy_for_prompt = []
            
            # The system message is the candidate prompt being evaluated.
            # The user message contains the questions to classify.
            messages = [
                {"role": "system", "content": prompt_text}, 
                {"role": "user", "content": f"Questions:\n\n{questions_list_str}\n\nNote: Ensure you respond with {current_batch_size} labels."}
            ]
            
            # `function_templates[1]` is "QnA_bot"
            # No need to modify tmp_function_template for Ollama here, send_query2gpt handles it
            
            prompt_attempt_answers = [] # Store answers from this attempt
            try:
                response_data = send_query2gpt(
                    self.client, messages, function_templates[1], 
                    temperature=0.0, pause=self.pause_value,
                    batch_size=current_batch_size, label_vocab=self.label_vocab_list, # Crucial for Ollama
                    retries=self.num_retries # Use retries from __init__
                )
                
                if response_data and "label_array" in response_data:
                    predicted_label_dicts = response_data['label_array']
                    
                    if len(predicted_label_dicts) == current_batch_size:
                        predicted_texts = [item.get('label', "ERROR_NO_LABEL") for item in predicted_label_dicts]
                        prompt_attempt_answers = predicted_texts # Store these
                        
                        correct_count = sum(1 if pred == actual else 0 for pred, actual in zip(predicted_texts, correct_answers_text))
                        accuracy = correct_count / current_batch_size if current_batch_size > 0 else 0.0
                        accumulated_accuracy_for_prompt.append(accuracy)
                    else:
                        # print(f"Warning (qa_optim eval): Prompt {prompt_idx+1} - Mismatch in QnA_bot output count. Expected {current_batch_size}, Got {len(predicted_label_dicts)}. Accuracy 0.")
                        accumulated_accuracy_for_prompt.append(0.0)
                        prompt_attempt_answers = ["ERROR_COUNT_MISMATCH"] * current_batch_size
                else:
                    # print(f"Warning (qa_optim eval): Prompt {prompt_idx+1} - No 'label_array' in QnA_bot response. Accuracy 0. Response: {response_data}")
                    accumulated_accuracy_for_prompt.append(0.0)
                    prompt_attempt_answers = ["ERROR_NO_LABEL_ARRAY"] * current_batch_size
            
            except Exception as e:
                # print(f"Error during fitness evaluation for Prompt {prompt_idx+1} ('{prompt_text[:30]}...'): {e}. Accuracy 0.")
                accumulated_accuracy_for_prompt.append(0.0)
                prompt_attempt_answers = [f"ERROR_EXCEPTION_{type(e).__name__}"] * current_batch_size

            all_prompt_answers_text.append(prompt_attempt_answers if prompt_attempt_answers else ["ERROR_NO_ANSWERS_RECORDED"]*current_batch_size)
            
            # Average accuracy for this prompt over its (single, in this simplified version) attempt
            final_prompt_fitness = np.mean(accumulated_accuracy_for_prompt) if accumulated_accuracy_for_prompt else 0.0
            all_fitness_scores.append(final_prompt_fitness)

        return all_fitness_scores, questions_for_eval_text, correct_answers_text, all_prompt_answers_text


    def select_top_prompts(self, fitness_scores, population, prompt_answers_list, top_fraction=0.5):
        if not population: # Handle empty population
            return [], []
        
        # Ensure prompt_answers_list has the same length as population for zipping
        # If prompt_answers_list was shorter (e.g., due to errors in eval), pad it.
        if len(prompt_answers_list) < len(population):
            # print(f"Warning (select_top): Mismatch len prompt_answers ({len(prompt_answers_list)}) vs population ({len(population)}). Padding.")
            padding_needed = len(population) - len(prompt_answers_list)
            # Get batch size from a valid entry or default
            batch_s = len(self.label_vocab_list) if self.label_vocab_list else 1
            if prompt_answers_list and prompt_answers_list[0]: batch_s = len(prompt_answers_list[0])

            default_answer_padding = [f"PAD_ERROR"] * batch_s
            prompt_answers_list.extend([default_answer_padding] * padding_needed)


        paired_list = list(zip(population, fitness_scores, prompt_answers_list))
        # Sort by fitness score (second element of tuple, x[1])
        sorted_prompts_data = sorted(paired_list, key=lambda x: x[1], reverse=True)
        
        cutoff = int(len(sorted_prompts_data) * top_fraction)
        if cutoff == 0 and len(sorted_prompts_data) > 0 : cutoff = 1 # Ensure at least one if possible
        
        top_prompts_text = [item[0] for item in sorted_prompts_data[:cutoff]]
        top_prompts_answers = [item[2] for item in sorted_prompts_data[:cutoff]]
        
        return top_prompts_text, top_prompts_answers


    def gpt_mutate(self, prompt_to_mutate): # Renamed from original to avoid conflict with list method
        # `function_templates[2]` is "prompt_mutate"
        # The description in function_templates[2] contains "Degree to modify: "
        # We need to append a random degree to it, or let send_query2gpt handle it if it's part of system message.
        # For now, assume send_query2gpt structure is enough, and the template description is static.
        
        # System message: Overarching problem description
        # User message: The prompt to mutate
        # function_templates[2]["description"] provides the core mutation instruction.
        
        system_message_content = (
            f"Note: For this task the over-arching Problem Description is: {self.problem_description}\n"
            f"The degree to modify the prompt (as a percentage from 0.0 to 1.0, e.g., 0.2 for 20% change) is: {round(random.random() * 0.3 + 0.1, 3)}" # Small mutation
        )
        # function_templates[2]["description"] will be added by send_query2gpt

        messages = [
            {"role": "system", "content": system_message_content},
            {"role": "user", "content": f"Modify the following prompt: \"\"\"{prompt_to_mutate}\"\"\""}
        ]
        try:
            response_data = send_query2gpt(self.client, messages, function_templates[2], 
                                           temperature=random.random()/2 + 0.5, # 0.5 to 1.0
                                           pause=self.pause_value, retries=self.num_retries)
            if response_data and "mutated_prompt" in response_data:
                return response_data['mutated_prompt']
            else:
                # print(f"Warning (gpt_mutate): Mutation failed. Falling back to original. Response: {response_data}")
                return prompt_to_mutate # Fallback
        except Exception as e:
            # print(f"Error during gpt_mutate with Ollama: {e}. Falling back to original.")
            return prompt_to_mutate


    def gpt_mix_and_match(self, template_prompt, additive_prompt, questions_context, correct_answers_context, answers_from_parents_context):
        # `function_templates[3]` is "prompt_crossover"
        # Prepare example string for context
        example_str_parts = []
        # Ensure all lists for zipping have the same length
        min_len = min(len(questions_context), len(correct_answers_context), 
                      len(answers_from_parents_context[0]) if answers_from_parents_context and answers_from_parents_context[0] else 0,
                      len(answers_from_parents_context[1]) if answers_from_parents_context and len(answers_from_parents_context) > 1 and answers_from_parents_context[1] else 0)
        
        num_examples = min(5, min_len) # Show up to 5 examples

        if num_examples > 0 and len(answers_from_parents_context) >= 2:
            for i in range(num_examples):
                q = questions_context[i]
                a = correct_answers_context[i]
                p0_ans = answers_from_parents_context[0][i]
                p1_ans = answers_from_parents_context[1][i]
                example_str_parts.append(f'Question: """{q}"""\nIdeal Answer: """{a}"""\nTemplate Parent\'s Answer: """{p0_ans}"""\nAdditive Parent\'s Answer: """{p1_ans}"""')
        
        example_context_str = "\n\n".join(example_str_parts) if example_str_parts else "No example context available."

        system_message_content = (
            f"Note: For this task the over-arching Problem Description is: {self.problem_description}\n\n"
            f"Example & History for context:\n{example_context_str}\n\n"
            "Note: You can use previous mistakes as stepping stones, to quote words/semantics/phrases/keywords/verbs which you think led to the mistake by the AI."
        )
        # function_templates[3]["description"] will be added by send_query2gpt

        messages = [
            {"role": "system", "content": system_message_content}, 
            {"role": "user", "content": f"Template Prompt: \"\"\"{template_prompt}\"\"\"\nAdditive Prompt: \"\"\"{additive_prompt}\"\"\""}
        ]
        try:
            response_data = send_query2gpt(self.client, messages, function_templates[3],
                                           temperature=0.7, pause=self.pause_value, retries=self.num_retries)
            if response_data and "child_prompt" in response_data:
                return response_data['child_prompt']
            else:
                # print(f"Warning (gpt_mix_and_match): Crossover failed. Falling back to template. Response: {response_data}")
                return template_prompt # Fallback
        except Exception as e:
            # print(f"Error during gpt_mix_and_match with Ollama: {e}. Falling back to template.")
            return template_prompt

    def crossover_using_gpt(self, top_prompts, questions_context, correct_answers_context, top_prompts_answers_context):
        if len(top_prompts) < 2:
            # print("Warning (crossover): Too few prompts to cross-over. Returning original top prompts or mutated versions.")
            # If only one prompt, mutate it or return as is.
            if len(top_prompts) == 1:
                return [self.gpt_mutate(top_prompts[0])]
            return top_prompts # Or [] if empty

        new_crossed_prompts = []
        # Iterate pairing prompts for crossover
        for i in range(0, len(top_prompts) -1, 2): # Ensure i+1 is always valid
            parent1_prompt = top_prompts[i]
            parent2_prompt = top_prompts[i+1]
            
            # Get corresponding answers for these parents if available
            parent_answers_for_context = []
            if top_prompts_answers_context and len(top_prompts_answers_context) > i+1:
                 parent_answers_for_context = [top_prompts_answers_context[i], top_prompts_answers_context[i+1]]
            else: # Fallback if answers are not perfectly aligned (should not happen with correct select_top_prompts)
                 # print("Warning (crossover): top_prompts_answers_context alignment issue. Using empty parent answers.")
                 dummy_batch_size = len(questions_context) if questions_context else 1
                 parent_answers_for_context = [["N/A"]*dummy_batch_size, ["N/A"]*dummy_batch_size]


            if parent1_prompt.lower().strip() == parent2_prompt.lower().strip():
                # print("Info (crossover): Parent prompts are identical. Mutating parent2 before crossover.")
                parent2_prompt = self.gpt_mutate(parent2_prompt) # Mutate if identical to encourage diversity
            
            child_prompt = self.gpt_mix_and_match(parent1_prompt, parent2_prompt, 
                                                  questions_context, correct_answers_context, 
                                                  parent_answers_for_context)
            new_crossed_prompts.append(child_prompt)

        # If odd number of top_prompts, the last one wasn't crossed. Carry it over or mutate.
        if len(top_prompts) % 2 == 1:
            last_prompt = top_prompts[-1]
            new_crossed_prompts.append(self.gpt_mutate(last_prompt)) # Mutate the leftover

        return new_crossed_prompts


    def mutate_prompts(self, prompts_to_mutate, mutation_rate=0.1): # Renamed from original
        final_mutated_prompts = []
        for p_text in prompts_to_mutate:
            if random.random() < mutation_rate:
                final_mutated_prompts.append(self.gpt_mutate(p_text))
            else:
                final_mutated_prompts.append(p_text)
        return final_mutated_prompts


    def genetic_algorithm(self, mutation_rate=0.1, elite_size_fraction=0.1, random_injection_fraction=0.1): # Added more GA params
        # Population size is now self.population_size_ga
        
        # print(f"Starting Genetic Algorithm with Ollama backend. Population: {self.population_size_ga}, Generations: {self.generations}")
        current_population_prompts = self.generate_init_prompts(n=self.population_size_ga) # Ensure initial pop matches size

        if not current_population_prompts:
            raise Exception("Failed to generate any initial prompts. Aborting genetic algorithm.")
        
        # Ensure initial population is of the correct size
        if len(current_population_prompts) < self.population_size_ga:
            # print(f"Warning: Initial population size {len(current_population_prompts)} is less than target {self.population_size_ga}. Padding...")
            if current_population_prompts: # if some prompts were generated
                current_population_prompts = (current_population_prompts * (self.population_size_ga // len(current_population_prompts) + 1))[:self.population_size_ga]
            else: # if no prompts were generated at all (should be caught by above)
                current_population_prompts = [f"Fallback prompt: Please classify. Labels: {self.label_vocab_list}"] * self.population_size_ga

        # Main GA loop
        # bar = tqdm(range(self.generations), desc="GA Generations (Ollama)")
        for gen_id in range(self.generations): # Using simple range for now, tqdm can be re-added
            # print(f"\n--- Generation {gen_id + 1}/{self.generations} ---")
            # print(f"Current Population ({len(current_population_prompts)} prompts):")
            # for i, p_text in enumerate(current_population_prompts): print(f"  P{i}: {p_text[:70]}...")

            # Evaluate fitness of the current population
            fitness_scores, questions_used_for_eval, correct_answers_for_eval, all_prompt_answers = self.evaluate_fitness(current_population_prompts)
            
            # Save epoch data
            df_epoch = pd.DataFrame({'Prompt': current_population_prompts, 'Fitness Score': fitness_scores})
            df_epoch.to_csv(os.path.join(self.run_path, f'epoch_{gen_id}.csv'), index=False)
            avg_fitness = np.mean(fitness_scores) if fitness_scores else 0.0
            max_fitness = np.max(fitness_scores) if fitness_scores else 0.0
            print(f"Generation {gen_id}: Avg Fitness = {avg_fitness:.4f}, Max Fitness = {max_fitness:.4f}")

            # Selection: select top N prompts (elitism + parents for crossover)
            # Top fraction for elitism and also for being parents in crossover.
            # `select_top_prompts` already implements a top_fraction selection.
            num_elites = int(self.population_size_ga * elite_size_fraction)
            if num_elites == 0 and self.population_size_ga > 0: num_elites = 1 # Ensure at least one elite

            # Select a larger pool for parents than just elites, then elites are just copied.
            # For simplicity, let's use top_fraction=0.5 for selecting parents for crossover.
            parents_for_crossover, parent_answers_for_crossover_context = self.select_top_prompts(
                fitness_scores, current_population_prompts, all_prompt_answers, top_fraction=0.5
            )
            
            # Elitism: Directly carry over the very best `num_elites` individuals
            sorted_indices = np.argsort(fitness_scores)[::-1] # Sort descending by fitness
            elite_prompts = [current_population_prompts[i] for i in sorted_indices[:num_elites]]
            
            next_generation_prompts = list(elite_prompts) # Start new generation with elites

            # Crossover: Generate offspring from the selected parents
            # `crossover_using_gpt` expects a list of prompts to pair up.
            if len(parents_for_crossover) >= 2:
                offspring_prompts = self.crossover_using_gpt(
                    parents_for_crossover, questions_used_for_eval, 
                    correct_answers_for_eval, parent_answers_for_crossover_context
                )
                next_generation_prompts.extend(offspring_prompts)
            # else:
                # print("Info: Not enough parents for crossover, skipping.")

            # Mutation: Mutate some prompts in the current new generation (excluding pure elites for now, or include them)
            # Let's mutate the non-elite part of the new generation.
            prompts_to_consider_for_mutation = next_generation_prompts[num_elites:]
            mutated_offspring = self.mutate_prompts(prompts_to_consider_for_mutation, mutation_rate)
            next_generation_prompts = elite_prompts + mutated_offspring # Reconstruct with elites intact

            # Random Injection: Add some completely new random prompts
            num_random_to_inject = int(self.population_size_ga * random_injection_fraction)
            if num_random_to_inject > 0:
                random_new_prompts = self.generate_init_prompts(n=num_random_to_inject) # Request specific number
                # Ensure random_new_prompts is exactly num_random_to_inject
                if len(random_new_prompts) < num_random_to_inject and random_new_prompts:
                    random_new_prompts = (random_new_prompts * (num_random_to_inject // len(random_new_prompts) + 1))[:num_random_to_inject]
                elif not random_new_prompts: # if generate_init_prompts failed for injection
                     random_new_prompts = [f"Fallback random injection: Classify with {self.label_vocab_list}"] * num_random_to_inject

                next_generation_prompts.extend(random_new_prompts)

            # Fill up population if needed (e.g. if crossover/mutation produced fewer than expected or random injection small)
            # Or truncate if too many. The goal is to maintain self.population_size_ga.
            if len(next_generation_prompts) > self.population_size_ga:
                # If too many, select the best ones based on some heuristic (e.g., random, or re-evaluate a subset)
                # For now, let's just truncate randomly, or better, from the "lesser" parts (e.g. end of list if elites are at start)
                current_population_prompts = next_generation_prompts[:self.population_size_ga]
            elif len(next_generation_prompts) < self.population_size_ga:
                # print(f"Info: Next gen size {len(next_generation_prompts)} < target {self.population_size_ga}. Filling with random/duplicates.")
                needed = self.population_size_ga - len(next_generation_prompts)
                fill_prompts = self.generate_init_prompts(n=needed)
                if len(fill_prompts) < needed and fill_prompts:
                     fill_prompts = (fill_prompts * (needed // len(fill_prompts) + 1))[:needed]
                elif not fill_prompts:
                     fill_prompts = [f"Fallback fill: Classify. Labels: {self.label_vocab_list}"] * needed
                next_generation_prompts.extend(fill_prompts)
                current_population_prompts = next_generation_prompts[:self.population_size_ga] # Ensure exact size
            else:
                current_population_prompts = next_generation_prompts

            # bar.set_description(f"Epoch {gen_id+1}: AvgAcc={avg_fitness*100:.1f}%")
        # bar.close()

        # Final evaluation to get the best prompt from the last population
        final_fitness_scores, _, _, _ = self.evaluate_fitness(current_population_prompts)
        best_prompt_idx = np.argmax(final_fitness_scores) if final_fitness_scores else 0
        
        print("\nGenetic Algorithm Completed.")
        if current_population_prompts and final_fitness_scores:
            print(f"Best prompt from final generation: '{current_population_prompts[best_prompt_idx][:100]}...' with fitness: {final_fitness_scores[best_prompt_idx]:.4f}")
        else:
            print("No prompts or fitness scores available at the end of GA.")

        return current_population_prompts