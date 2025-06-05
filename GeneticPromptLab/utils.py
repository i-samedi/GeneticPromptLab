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

import json
import time
import paramiko
import os
import shlex

# Load SSH configuration globally for this module
# Assumes ssh_config.json is in the project root directory (parent of GeneticPromptLab directory)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SSH_CONFIG_PATH = os.path.join(PROJECT_ROOT, 'ssh_config.json')

if not os.path.exists(SSH_CONFIG_PATH):
    raise FileNotFoundError(f"SSH configuration file not found at: {SSH_CONFIG_PATH}. "
                            "Please create ssh_config.json in the project root (e.g., GeneticPromptLab/../ssh_config.json).")
with open(SSH_CONFIG_PATH, "r") as f:
    ssh_config = json.load(f)

def execute_remote_ollama_command(gateway_ssh_client, ollama_command_on_colossus):
    """
    Executes a command on Colossus via the gateway SSH client.
    """
    colossus_user = ssh_config['colossus']['username']
    colossus_host = ssh_config['colossus']['hostname']
    colossus_pass = ssh_config['colossus']['password']

    escaped_ollama_cmd = shlex.quote(ollama_command_on_colossus)
    # Using sshpass. Ensure sshpass is installed on the gateway machine.
    remote_command_on_gateway = f"sshpass -p '{colossus_pass}' ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null {colossus_user}@{colossus_host} {escaped_ollama_cmd}"
    
    # print(f"DEBUG (utils): Executing on gateway: {remote_command_on_gateway}")

    stdin, stdout, stderr = gateway_ssh_client.exec_command(remote_command_on_gateway, timeout=180) # Increased timeout
    exit_status = stdout.channel.recv_exit_status() 

    stdout_output = stdout.read().decode('utf-8').strip()
    stderr_output = stderr.read().decode('utf-8').strip()

    if exit_status != 0:
        # print(f"DEBUG (utils): Remote command stdout: {stdout_output}")
        # print(f"DEBUG (utils): Remote command stderr: {stderr_output}")
        raise Exception(f"Error executing command on Colossus (via gateway). Exit status: {exit_status}. Stderr: {stderr_output}. Command: {remote_command_on_gateway}")
    
    return stdout_output

def send_query2gpt(client, messages, function_template, temperature=0, pause=5, batch_size=None, label_vocab=None, retries=1):
    """
    Sends a query to Ollama on the remote Colossus server via SSH.
    'client' parameter is kept for signature compatibility but is NOT used for Ollama.
    'function_template' (a dict) guides the JSON output structure via its 'name'.
    'batch_size' and 'label_vocab' are needed for 'QnA_bot' template.
    'retries' from calling function (e.g. qa_optim's num_retries).
    """
    # print(f"DEBUG (utils): send_query2gpt called with function_template name: {function_template.get('name')}, pause: {pause}, retries: {retries}")

    system_prompt_content = ""
    user_prompt_content = ""
    # The 'messages' from qa_optim often have the core instruction in system and data in user role.
    # However, function_templates also have a "description" which often acts as a primary system instruction.
    # We need to combine these carefully.

    # Let's prioritize the function_template's description as the main system-level instruction for Ollama,
    # and treat 'messages' content as supplementary or user-level input.

    for msg in messages:
        if msg['role'] == 'system':
            # This could be problem description or a candidate prompt.
            system_prompt_content += msg['content'] + "\n\n"
        elif msg['role'] == 'user':
            user_prompt_content = msg['content'] # Typically the data (questions, examples)

    json_structure_description = ""
    template_name = function_template.get("name")
    expected_key = None # The main key expected in the returned JSON from the model

    if template_name == "generate_prompts": # from function_templates[0]
        expected_key = "prompt"
        json_structure_description = f"""
You MUST respond with ONLY a single well-formed JSON object.
The JSON object must have a single key: "{expected_key}".
The value of "{expected_key}" must be a string containing the system prompt you generated.
Example:
{{
  "{expected_key}": "You are a helpful assistant for text classification..."
}}
Do not include any other text, explanations, or apologies before or after the JSON object.
"""
    elif template_name == "QnA_bot": # from function_templates[1]
        expected_key = "label_array"
        if batch_size is None or label_vocab is None:
            raise ValueError("batch_size and label_vocab must be provided for 'QnA_bot' template.")
        json_structure_description = f"""
You MUST respond with ONLY a single well-formed JSON object.
The JSON object must have a single key: "{expected_key}".
The value of "{expected_key}" must be a list of JSON objects.
There should be exactly {batch_size} objects in the "{expected_key}" list.
Each object in the "{expected_key}" list must have a single key: "label".
The value for "label" MUST be one of the following strings: {json.dumps(label_vocab)}.
Example for batch_size=2 and labels ["X", "Y"]:
{{
  "{expected_key}": [
    {{ "label": "X" }},
    {{ "label": "Y" }}
  ]
}}
Do not include any other text, explanations, or apologies before or after the JSON object.
"""
    elif template_name == "prompt_mutate": # from function_templates[2]
        expected_key = "mutated_prompt"
        json_structure_description = f"""
You MUST respond with ONLY a single well-formed JSON object.
The JSON object must have a single key: "{expected_key}".
The value of "{expected_key}" must be a string containing the mutated prompt.
Example:
{{
  "{expected_key}": "This is a slightly changed version of the original prompt..."
}}
Do not include any other text, explanations, or apologies before or after the JSON object.
"""
    elif template_name == "prompt_crossover": # from function_templates[3]
        expected_key = "child_prompt"
        json_structure_description = f"""
You MUST respond with ONLY a single well-formed JSON object.
The JSON object must have a single key: "{expected_key}".
The value of "{expected_key}" must be a string containing the new child prompt.
Example:
{{
  "{expected_key}": "This is a new prompt created by combining two parent prompts..."
}}
Do not include any other text, explanations, or apologies before or after the JSON object.
"""
    else:
        raise ValueError(f"Unknown function_template name: {template_name}. Supported: 'generate_prompts', 'QnA_bot', 'prompt_mutate', 'prompt_crossover'")

    # Combine function_template's own description (often the meta-instruction)
    # with the system content from `messages` (e.g., problem description, candidate prompt)
    # and finally our JSON formatting instruction.
    
    # The main instruction for the LLM often comes from function_template["description"]
    # and the `system_prompt_content` (from `messages`) provides context or the thing to operate on.
    final_system_message_parts = []
    if function_template.get("description"):
        final_system_message_parts.append(function_template["description"])
    if system_prompt_content: # This could be the "Problem Description" or a prompt to be mutated/crossovered
        final_system_message_parts.append(system_prompt_content.strip())
    
    final_system_message_parts.append(json_structure_description)
    final_system_prompt = "\n\n".join(final_system_message_parts)

    ollama_messages = [{"role": "system", "content": final_system_prompt}]
    if user_prompt_content: # This is the data (questions, examples, prompts to combine)
         ollama_messages.append({"role": "user", "content": user_prompt_content})

    ollama_payload = {
        "model": ssh_config['ollama_model'],
        "messages": ollama_messages,
        "format": "json",
        "stream": False,
        "options": {
            "temperature": float(temperature), # Ensure temperature is float
            # "seed": 0 # Consider for reproducibility if supported by your Ollama version/model
        }
    }
    
    payload_str_for_curl = json.dumps(ollama_payload)
    # Added connect and max-time timeouts for curl
    ollama_command_on_colossus = f"curl -s --connect-timeout 60 --max-time 180 http://localhost:11434/api/chat -d '{payload_str_for_curl}'"

    last_exception = None
    # The 'retries' parameter here is the number of *additional* attempts after the first one.
    # So, loop `retries + 1` times.
    for attempt in range(retries + 1):
        gateway_client = None # Initialize to None for finally block
        try:
            gateway_client = paramiko.SSHClient()
            gateway_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            # print(f"DEBUG (utils): Connecting to gateway (attempt {attempt + 1}/{retries + 1})...")
            gateway_client.connect(
                hostname=ssh_config['gateway']['hostname'],
                port=ssh_config['gateway']['port'],
                username=ssh_config['gateway']['username'],
                password=ssh_config['gateway']['password'],
                timeout=60 # Connection timeout
            )
            # print(f"DEBUG (utils): Connected to gateway. Executing Ollama command...")
            raw_response = execute_remote_ollama_command(gateway_client, ollama_command_on_colossus)
            # print(f"DEBUG (utils): Raw Ollama response string (attempt {attempt+1}): {raw_response}")

            ollama_api_response = json.loads(raw_response)
            model_generated_json_str = ollama_api_response.get("message", {}).get("content", "")
            
            if not model_generated_json_str:
                error_detail = f"Ollama response 'message.content' is empty. Full API response: {ollama_api_response}"
                if ollama_api_response.get("error"):
                    error_detail = f"Ollama API error: {ollama_api_response.get('error')}"
                # print(f"DEBUG (utils): {error_detail}")
                raise ValueError(error_detail)

            # print(f"DEBUG (utils): Model generated JSON string (attempt {attempt+1}): {model_generated_json_str}")
            parsed_answer = json.loads(model_generated_json_str) # This should be the JSON with expected_key
            
            if expected_key not in parsed_answer:
                raise ValueError(f"Ollama response for '{template_name}' missing expected key '{expected_key}'. Got: {parsed_answer}")

            time.sleep(float(pause)) # Ensure pause is float
            return parsed_answer # This is the dict like {"prompt": "..."} or {"label_array": [...]}

        except paramiko.ssh_exception.SSHException as e:
            last_exception = e
            print(f"SSH connection error (attempt {attempt + 1}/{retries + 1}): {e}")
        except json.JSONDecodeError as e:
            last_exception = e
            current_raw_response = raw_response if 'raw_response' in locals() else "N/A (raw_response not captured)"
            current_model_json = model_generated_json_str if 'model_generated_json_str' in locals() else "N/A (model_json not captured)"
            print(f"JSON decoding error (attempt {attempt + 1}/{retries + 1}): {e}. "
                  f"Model JSON str: '{current_model_json}'. Raw API response: '{current_raw_response}'")
        except Exception as e: # Catch broader exceptions including ValueErrors from validation
            last_exception = e
            print(f"General error in send_query2gpt (attempt {attempt + 1}/{retries + 1}) for {template_name}: {e}")
        finally:
            if gateway_client:
                gateway_client.close()
                # print("DEBUG (utils): Gateway connection closed.")
        
        if attempt < retries:
            # print(f"Retrying in {pause * (attempt + 2)} seconds...") # Slightly increasing backoff
            time.sleep(float(pause) * (attempt + 2))
        elif last_exception: # If all retries failed, raise the last known exception
            print(f"Max retries ({retries}) reached for query {template_name}. Last error: {last_exception}")
            raise last_exception
        else: # Should not happen if an exception was always set
            raise Exception(f"send_query2gpt failed for {template_name} after {retries+1} attempts without a specific exception recorded.")
            
    # This part should ideally not be reached if exceptions are handled correctly.
    # Return an empty dict or raise error if all retries fail.
    print(f"CRITICAL (utils): send_query2gpt exhausted retries and did not raise an exception. This is a bug.")
    return {} 
# --- END OF FILE GeneticPromptLab/utils.py ---