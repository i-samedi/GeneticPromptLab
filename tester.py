from GeneticPromptLab.function_templates import function_templates
import pandas as pd
import numpy as np
import json
import os
import time
from tqdm import trange
#from openai import OpenAI
import paramiko

with open("ssh_config.json", "r") as f:
    ssh_config = json.load(f)

qa_function_template = function_templates[1]

# with open("openai_api.key", "r") as f:
#     key = f.read()
# client = OpenAI(api_key=key.strip())

def read_data(path_test_df, path_label_dict):
    with open(path_label_dict, "r") as f:
        label_dict = json.load(f)
    df = pd.read_csv(path_test_df)
    questions = df['question'].tolist()
    # answers = [label_dict[v] for v in df['label'].tolist()]
    answers = [label_dict[str(v)] if isinstance(v, int) else label_dict[v] for v in df['label'].tolist()]
    return questions, answers, label_dict

def read_latest_epoch_data(run_id):
    dir_path = f"./runs/{run_id}/"
    files = [f for f in os.listdir(dir_path) if f.startswith('epoch_') and f.endswith('.csv')]
    # Sort files based on numerical value of epoch_id
    if not files:
        raise FileNotFoundError(f"No epoch CSV files found in {dir_path}")
    sorted_files = sorted(files, key=lambda x: int(x.split('_')[1].split('.')[0]))
    latest_file_path = os.path.join(dir_path, sorted_files[-1])
    df = pd.read_csv(latest_file_path)
    return df

def get_highest_fitness_prompt(df):
    max_fitness_row = df[df['Fitness Score'] == df['Fitness Score'].max()]
    if max_fitness_row.empty:
        raise ValueError("No rows found with maximum fitness score, or DataFrame is empty.")
    highest_fitness_prompt = max_fitness_row['Prompt'].iloc[0]
    return highest_fitness_prompt

# def send_query2gpt(client, messages, function_template, temperature=0, pause=5):
#     response = client.chat.completions.create(
#         model="gpt-3.5-turbo",
#         messages=messages,
#         temperature=temperature,
#         max_tokens=512,
#         functions=[function_template], 
#         seed=0,
#         function_call={"name": function_template["name"]}
#     )
#     answer = response.choices[0].message.function_call.arguments
#     generated_response = json.loads(answer)
#     time.sleep(pause)
#     return generated_response

def execute_remote_ollama_command(gateway_ssh_client, ollama_command_on_colossus):
    """
    Executes a command on Colossus via the gateway SSH client.
    The command to Colossus itself includes an SSH login to Colossus.
    """
    colossus_user = ssh_config['colossus']['username']
    colossus_host = ssh_config['colossus']['hostname']
    colossus_pass = ssh_config['colossus']['password']
    
    import shlex
    escaped_ollama_cmd = shlex.quote(ollama_command_on_colossus)
    # We need to ensure the password is also handled carefully if it contains special chars.
    # For this example, assuming simple password.
    remote_command_on_gateway = f"sshpass -p '{colossus_pass}' ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null {colossus_user}@{colossus_host} {escaped_ollama_cmd}"
    
    print(f"DEBUG: Executing on gateway: {remote_command_on_gateway}") # For debugging

    stdin, stdout, stderr = gateway_ssh_client.exec_command(remote_command_on_gateway)
    exit_status = stdout.channel.recv_exit_status() # Wait for command to complete

    stdout_output = stdout.read().decode('utf-8').strip()
    stderr_output = stderr.read().decode('utf-8').strip()

    if exit_status != 0:
        # print(f"DEBUG: Remote command stdout: {stdout_output}") # For debugging
        # print(f"DEBUG: Remote command stderr: {stderr_output}") # For debugging
        raise Exception(f"Error executing command on Colossus (via gateway). Exit status: {exit_status}. Stderr: {stderr_output}")
    
    return stdout_output

def send_query_to_ollama(system_prompt_content, user_prompt_content, label_vocab, batch_size, temperature=0, pause=5):
    """
    Sends a query to Ollama on the remote Colossus server via SSH.
    """
    # Construct the prompt for Ollama to output JSON
    # The 'qa_function_template' structure guides this.
    # We need 'label_array' with 'label' items.
    
    json_structure_description = f"""
You MUST respond with ONLY a single well-formed JSON object.
The JSON object must have a single key: "label_array".
The value of "label_array" must be a list of JSON objects.
There should be exactly {batch_size} objects in the "label_array" list.
Each object in the "label_array" list must have a single key: "label".
The value for "label" MUST be one of the following strings: {json.dumps(label_vocab)}.
Example for batch_size=2 and labels ["X", "Y"]:
{{
  "label_array": [
    {{ "label": "X" }},
    {{ "label": "Y" }}
  ]
}}
Do not include any other text, explanations, or apologies before or after the JSON object.
"""

    full_prompt = f"{system_prompt_content}\n\n{json_structure_description}\n\nUser Question(s):\n{user_prompt_content}"

    # Prepare the Ollama API payload for the /api/chat endpoint
    # Note: Ollama's /api/generate might be simpler if you just have one big prompt.
    # /api/chat is more aligned with OpenAI's message structure.
    ollama_payload = {
        "model": ssh_config['ollama_model'],
        "messages": [
            {"role": "system", "content": system_prompt_content + "\n\n" + json_structure_description}, # Combine system instructions
            {"role": "user", "content": user_prompt_content}
        ],
        "format": "json", # Request JSON output from Ollama
        "stream": False,
        "options": { # Options equivalent to temperature, seed for some models
            "temperature": temperature
            # Ollama's seed support varies by model and version
        }
    }
    
    # Escape the JSON payload for shell command
    # Using single quotes around the JSON data for curl
    payload_str_for_curl = json.dumps(ollama_payload)

    # Command to be executed on Colossus
    # curl -s to suppress progress meter
    ollama_command_on_colossus = f"curl -s http://localhost:11434/api/chat -d '{payload_str_for_curl}'"

    # Establish SSH connection to the gateway
    gateway_client = paramiko.SSHClient()
    gateway_client.set_missing_host_key_policy(paramiko.AutoAddPolicy()) # Not recommended for production

    try:
        gateway_client.connect(
            hostname=ssh_config['gateway']['hostname'],
            port=ssh_config['gateway']['port'],
            username=ssh_config['gateway']['username'],
            password=ssh_config['gateway']['password'],
            timeout=20 # Add a timeout
        )

        # print("DEBUG: Connected to gateway.") # For debugging
        raw_response = execute_remote_ollama_command(gateway_client, ollama_command_on_colossus)
        # print(f"DEBUG: Raw Ollama response string: {raw_response}") # For debugging

    finally:
        gateway_client.close()
        # print("DEBUG: Gateway connection closed.") # For debugging

    time.sleep(pause)

    try:
        # Ollama with format: "json" using /api/chat should return a JSON where
        # the actual model's JSON response is in response_data['message']['content']
        ollama_api_response = json.loads(raw_response)
        # The content field should itself be a JSON string, as requested by "format": "json"
        model_generated_json_str = ollama_api_response.get("message", {}).get("content", "")
        if not model_generated_json_str:
             raise ValueError(f"Ollama response 'message.content' is empty. Full API response: {ollama_api_response}")
        
        # print(f"DEBUG: Model generated JSON string: {model_generated_json_str}") # For debugging
        parsed_answer = json.loads(model_generated_json_str) # This should be our target JSON structure

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from Ollama response: {e}")
        print(f"Raw response was: {raw_response}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred parsing Ollama response: {e}")
        print(f"Raw response was: {raw_response}")
        raise
        
    return parsed_answer

# def ag_news():
#     run_id = "XrFnn68pnF"
#     path_test_df = "./data/ag_news_test.csv"
#     path_label_dict = "./data/ag_news_label_dict.json"
#     questions, answers, label_vocab = read_data(path_test_df=path_test_df, path_label_dict=path_label_dict)
#     best_prompt = get_highest_fitness_prompt(read_latest_epoch_data(run_id))
#     batch_size = 10
#     qa_function_template["parameters"]["properties"]["label_array"]["items"]["properties"]["label"]["enum"] = label_vocab
#     qa_function_template["parameters"]["properties"]["label_array"]["items"]["properties"]["label"]["description"] += str(label_vocab)
#     qa_function_template["parameters"]["properties"]["label_array"]["minItems"] = batch_size
#     qa_function_template["parameters"]["properties"]["label_array"]["maxItems"] = batch_size
#     aggregate_accuracy = []
#     batches_skipped_count = 0
#     for i in trange(0, len(questions), batch_size, desc="testing_agnews"):
#         question_subset = questions[i:i+batch_size]
#         answer_subset = answers[i:i+batch_size]
#         questions_list = "\n\n".join([str(i+1)+'. """'+question+'"""' for i,question in enumerate(question_subset)])
#         try:
#             response = [v['label'] for v in send_query2gpt(client, [{"role": "system", "content": best_prompt}, {"role": "user", "content": "Questions:\n"+questions_list}], qa_function_template, temperature=0.0, pause=5)['label_array']]
#             accuracy = sum(1 if a == b else 0 for a, b in zip(response, answer_subset)) / len(response)
#             aggregate_accuracy.append(accuracy)
#         except:
#             batches_skipped_count += 1
#     print("Batches skipped", batches_skipped_count)
#     print("Accuracy:", str(round(100*np.mean(accuracy), 3))+"%")

def ag_news():
    run_id = "XrFnn68pnF" # Make sure this run_id exists and has data
    path_test_df = "./data/ag_news_test.csv"
    path_label_dict = "./data/ag_news_label_dict.json"
    
    # Ensure data directory and files exist
    if not os.path.exists(path_test_df) or not os.path.exists(path_label_dict):
        print(f"Missing data files for AG News. Run example_data_setup.py first.")
        print(f"Ensure a run directory for {run_id} (e.g., ./runs/{run_id}/epoch_0.csv) also exists.")
        return

    questions, answers, label_dict_map = read_data(path_test_df=path_test_df, path_label_dict=path_label_dict)
    label_vocab = list(label_dict_map.values()) # Get the list of label strings

    # Check if run directory and epoch files exist
    try:
        latest_epoch_df = read_latest_epoch_data(run_id)
        best_prompt = get_highest_fitness_prompt(latest_epoch_df)
    except FileNotFoundError as e:
        print(f"Error: {e}. Cannot proceed with ag_news test.")
        print(f"Please ensure that the run '{run_id}' has been executed and CSV files are present in './runs/{run_id}/'.")
        return
    except ValueError as e:
        print(f"Error: {e}. Check the contents of epoch files in './runs/{run_id}/'.")
        return


    batch_size = 10 # Keep batch size manageable for testing
    # qa_function_template modification is no longer for OpenAI, but guides prompt construction.
    # The actual "enum" and "minItems/maxItems" will be part of the text prompt to Ollama.

    aggregate_accuracy = []
    batches_skipped_count = 0
    
    # For testing, let's limit the number of questions
    num_test_questions = min(len(questions), 20) # Test with first 20 questions or fewer
    
    print(f"Testing AG News with a subset of {num_test_questions} questions.")

    for i in trange(0, num_test_questions, batch_size, desc="testing_agnews_ollama"):
        question_subset = questions[i:i+batch_size]
        answer_subset = answers[i:i+batch_size]
        
        # Adjust batch size if it's the last partial batch
        current_batch_size = len(question_subset)
        if current_batch_size == 0:
            continue

        questions_list_str = "\n\n".join([f"{idx+1}. \"\"\"{q}\"\"\"" for idx, q in enumerate(question_subset)])
        
        try:
            # Call the new Ollama function
            response_json = send_query_to_ollama(
                system_prompt_content=best_prompt,
                user_prompt_content="Classify the following questions:\n" + questions_list_str,
                label_vocab=label_vocab,
                batch_size=current_batch_size, # Use current batch size for JSON instruction
                temperature=0.0,
                pause=5 # Ollama might be slower or rate limits might not be an issue
            )
            
            # Extract labels from the response_json structured as per our prompt
            # Expected: {"label_array": [{"label": "World"}, {"label": "Sports"}, ...]}
            predicted_labels = [item['label'] for item in response_json.get('label_array', [])]

            if len(predicted_labels) != len(answer_subset):
                print(f"Warning: Mismatch in number of predictions ({len(predicted_labels)}) vs answers ({len(answer_subset)}). Skipping batch.")
                print(f"Response JSON: {response_json}")
                batches_skipped_count += 1
                continue
            
            accuracy = sum(1 if pred == actual else 0 for pred, actual in zip(predicted_labels, answer_subset)) / len(predicted_labels)
            aggregate_accuracy.append(accuracy)
        except Exception as e:
            print(f"Error processing batch {i//batch_size}: {e}")
            batches_skipped_count += 1
    
    print("Batches skipped:", batches_skipped_count)
    if aggregate_accuracy:
        mean_accuracy = np.mean(aggregate_accuracy) # Use aggregate_accuracy here
        print("Accuracy:", f"{round(100 * mean_accuracy, 3)}%")
    else:
        print("No batches were successfully processed to calculate accuracy.")

# def trec():
#     run_id = "08zLX4cd97"
#     path_test_df = "./data/trec_test.csv"
#     path_label_dict = "./data/trec_label_dict.json"
#     questions, answers, label_vocab = read_data(path_test_df=path_test_df, path_label_dict=path_label_dict)
#     best_prompt = get_highest_fitness_prompt(read_latest_epoch_data(run_id))
#     batch_size = 10
#     qa_function_template["parameters"]["properties"]["label_array"]["items"]["properties"]["label"]["enum"] = label_vocab
#     qa_function_template["parameters"]["properties"]["label_array"]["items"]["properties"]["label"]["description"] += str(label_vocab)
#     qa_function_template["parameters"]["properties"]["label_array"]["minItems"] = batch_size
#     qa_function_template["parameters"]["properties"]["label_array"]["maxItems"] = batch_size
#     aggregate_accuracy = []
#     batches_skipped_count = 0
#     for i in trange(0, len(questions), batch_size, desc="testing_agnews"):
#         question_subset = questions[i:i+batch_size]
#         answer_subset = answers[i:i+batch_size]
#         questions_list = "\n\n".join([str(i+1)+'. """'+question+'"""' for i,question in enumerate(question_subset)])
#         try:
#             response = [v['label'] for v in send_query2gpt(client, [{"role": "system", "content": best_prompt}, {"role": "user", "content": "Questions:\n"+questions_list}], qa_function_template, temperature=0.0, pause=5)['label_array']]
#             accuracy = sum(1 if a == b else 0 for a, b in zip(response, answer_subset)) / len(response)
#             aggregate_accuracy.append(accuracy)
#         except:
#             batches_skipped_count += 1
#     print("Batches skipped", batches_skipped_count)
#     print("Accuracy:", str(round(100*np.mean(aggregate_accuracy), 3))+"%")

def trec():
    run_id = "08zLX4cd97" # Make sure this run_id exists and has data
    path_test_df = "./data/trec_test.csv"
    path_label_dict = "./data/trec_label_dict.json"

    if not os.path.exists(path_test_df) or not os.path.exists(path_label_dict):
        print(f"Missing data files for TREC. Run example_data_setup.py first.")
        print(f"Ensure a run directory for {run_id} (e.g., ./runs/{run_id}/epoch_0.csv) also exists.")
        return

    questions, answers, label_dict_map = read_data(path_test_df=path_test_df, path_label_dict=path_label_dict)
    label_vocab = list(label_dict_map.values())

    try:
        latest_epoch_df = read_latest_epoch_data(run_id)
        best_prompt = get_highest_fitness_prompt(latest_epoch_df)
    except FileNotFoundError as e:
        print(f"Error: {e}. Cannot proceed with TREC test.")
        print(f"Please ensure that the run '{run_id}' has been executed and CSV files are present in './runs/{run_id}/'.")
        return
    except ValueError as e:
        print(f"Error: {e}. Check the contents of epoch files in './runs/{run_id}/'.")
        return

    batch_size = 10
    aggregate_accuracy = []
    batches_skipped_count = 0

    num_test_questions = min(len(questions), 20) # Test with first 20 questions or fewer
    print(f"Testing TREC with a subset of {num_test_questions} questions.")

    for i in trange(0, num_test_questions, batch_size, desc="testing_trec_ollama"):
        question_subset = questions[i:i+batch_size]
        answer_subset = answers[i:i+batch_size]
        current_batch_size = len(question_subset)
        if current_batch_size == 0:
            continue
            
        questions_list_str = "\n\n".join([f"{idx+1}. \"\"\"{q}\"\"\"" for idx, q in enumerate(question_subset)])
        
        try:
            response_json = send_query_to_ollama(
                system_prompt_content=best_prompt,
                user_prompt_content="Classify the following questions:\n" + questions_list_str,
                label_vocab=label_vocab,
                batch_size=current_batch_size,
                temperature=0.0,
                pause=5
            )
            predicted_labels = [item['label'] for item in response_json.get('label_array', [])]

            if len(predicted_labels) != len(answer_subset):
                print(f"Warning: Mismatch in number of predictions ({len(predicted_labels)}) vs answers ({len(answer_subset)}). Skipping batch.")
                batches_skipped_count += 1
                continue

            accuracy = sum(1 if pred == actual else 0 for pred, actual in zip(predicted_labels, answer_subset)) / len(predicted_labels)
            aggregate_accuracy.append(accuracy)
        except Exception as e:
            print(f"Error processing batch {i//batch_size}: {e}")
            batches_skipped_count += 1
            
    print("Batches skipped:", batches_skipped_count)
    if aggregate_accuracy:
        mean_accuracy = np.mean(aggregate_accuracy)
        print("Accuracy:", f"{round(100 * mean_accuracy, 3)}%")
    else:
        print("No batches were successfully processed to calculate accuracy.")

def main():
    ag_news()
    # trec()

if __name__=='__main__':
    # Ensure data files are present
    if not (os.path.exists('./data/ag_news_test.csv') and \
            os.path.exists('./data/ag_news_label_dict.json') and \
            os.path.exists('./data/trec_test.csv') and \
            os.path.exists('./data/trec_label_dict.json')):
        print("Data files not found. Please run `python example_data_setup.py` first.")
    else:
        # Ensure run directories exist for the specified run_ids
        # You would need to run the GeneticPromptLab main experiments first to generate these.
        # For now, we'll proceed, and functions will print errors if run data is missing.
        if not os.path.exists('./runs/XrFnn68pnF'):
            print("Warning: Run directory ./runs/XrFnn68pnF not found. AG News test might fail to load best prompt.")
            # As a fallback for testing the SSH connection, you could use a default prompt:
            # Create dummy epoch file if needed for testing, or modify `get_highest_fitness_prompt`
            # For now, relying on existing run data.
        if not os.path.exists('./runs/08zLX4cd97'):
            print("Warning: Run directory ./runs/08zLX4cd97 not found. TREC test might fail to load best prompt.")
        main()