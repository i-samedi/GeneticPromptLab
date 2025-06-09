# --- START OF FILE GeneticPromptLab/ollama_ssh_client.py ---

import paramiko
import json
import time
import sys
import select
import re

class OllamaSSHClient:
    """
    Gestiona una conexión SSH a través de un host de salto para interactuar
    con un modelo de Ollama en una shell interactiva.
    """
    def __init__(self, config_path='ssh_config.json'):
        # El constructor __init__ no cambia.
        print("[Ollama Client] Inicializando cliente SSH para Ollama...")
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            gateway_config = config['gateway']
            colossus_config = config['colossus']
            self.ollama_model = config['ollama_model']
        except Exception as e:
            print(f"[ERROR] No se pudo cargar la configuración desde {config_path}: {e}")
            sys.exit(1)
        self.gateway_client = None
        self.colossus_client = None
        self.shell = None
        try:
            print(f"[*] Conectando al gateway {gateway_config['hostname']}...")
            self.gateway_client = paramiko.SSHClient()
            self.gateway_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            self.gateway_client.connect(
                hostname=gateway_config['hostname'],
                port=gateway_config['port'],
                username=gateway_config['username'],
                password=gateway_config['password'],
                timeout=15
            )
            transport = self.gateway_client.get_transport()
            dest_addr = (colossus_config['hostname'], 22)
            local_addr = ('localhost', 0)
            proxy_channel = transport.open_channel("direct-tcpip", dest_addr, local_addr)
            print(f"[*] Conectando a {colossus_config['hostname']} a través del túnel...")
            self.colossus_client = paramiko.SSHClient()
            self.colossus_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            self.colossus_client.connect(
                hostname=colossus_config['hostname'],
                username=colossus_config['username'],
                password=colossus_config['password'],
                sock=proxy_channel
            )
            print("[+] Conexión a Colossus establecida.")
            self.shell = self.colossus_client.invoke_shell()
            self._read_interactive_output(timeout=2) 
            ollama_command = f"ollama run {self.ollama_model}\n"
            print(f"[*] Iniciando Ollama con el modelo: {self.ollama_model}...")
            self.shell.send(ollama_command)
            self._read_interactive_output(stop_string=">>>", timeout=20)
            print("[+] Ollama está listo para recibir prompts.")
        except Exception as e:
            print(f"[ERROR] Falló la inicialización del cliente SSH: {e}")
            self.close()
            sys.exit(1)

    def _clean_ansi_codes(self, text):
        ansi_escape_pattern = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
        text = ansi_escape_pattern.sub('', text)
        spinners = ['⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏', '⠋']
        for char in spinners:
            text = text.replace(char, '')
        return text

    def _read_interactive_output(self, stop_string=None, timeout=5.0):
        output_buffer = ""
        last_data_time = time.time()
        while time.time() - last_data_time < timeout:
            r, _, _ = select.select([self.shell], [], [], 0.1)
            if r and self.shell.recv_ready():
                chunk = self.shell.recv(4096).decode('utf-8', errors='ignore')
                if not chunk: break
                output_buffer += chunk
                last_data_time = time.time()
                if stop_string and self._clean_ansi_codes(output_buffer).strip().endswith(stop_string):
                    break
        return output_buffer
    
    def _find_matching_brace(self, text, start_index):
        if start_index >= len(text) or text[start_index] != '{':
            return -1
        open_braces = 1
        for i in range(start_index + 1, len(text)):
            if text[i] == '{':
                open_braces += 1
            elif text[i] == '}':
                open_braces -= 1
            if open_braces == 0:
                return i
        return -1

    def run_prompt_and_get_json(self, messages, function_template):
        system_prompt = next((m['content'] for m in messages if m['role'] == 'system'), "")
        user_prompt = next((m['content'] for m in messages if m['role'] == 'user'), "")
        
        # --- ESTRATEGIA FINAL: Pedir directamente el objeto de argumentos ---
        properties = function_template.get("parameters", {}).get("properties", {})
        example_obj = {}
        for key, prop_schema in properties.items():
            if prop_schema.get("type") == "string":
                example_obj[key] = "An example string value."
            elif prop_schema.get("type") == "array":
                example_obj[key] = [{"label": "Example Label 1"}] # Simplificado a un solo ejemplo
            else:
                example_obj[key] = "some value"
        
        json_example = json.dumps(example_obj)

        # --- INGENIERÍA DE PROMPTS FINAL ---
        full_prompt = (
            f"You are a JSON generating robot. You will be given a system instruction and a user request. "
            f"Your task is to generate a single, valid JSON object that fulfills the request. "
            f"System Instruction: '{system_prompt}'. "
            f"User Request: '{user_prompt}'. "
            f"Your response MUST be ONLY the JSON object, formatted exactly like this example: {json_example}. "
            f"Do not add any explanations or markdown. Your response must start with '{{' and end with '}}'. Begin:\n"
        )
        
        full_prompt = full_prompt.replace('\n', ' ').strip() + '\n'

        print("[*] Enviando prompt a Ollama...")
        self.shell.send(full_prompt)
        
        # Aumentamos el timeout y agregamos más logging
        print("[*] Esperando respuesta de Ollama (timeout: 60s)...")
        raw_output = self._read_interactive_output(timeout=60.0)
        
        if not raw_output:
            print("[ERROR] No se recibió respuesta de Ollama dentro del timeout")
            raise TimeoutError("No response from Ollama within timeout period")

        # --- LÓGICA DE EXTRACCIÓN Y REPARACIÓN ---
        clean_output = self._clean_ansi_codes(raw_output)
        print(f"[DEBUG] Respuesta limpia recibida: {clean_output[:200]}...")
        
        start_brace_index = clean_output.rfind('{')
        if start_brace_index == -1:
            print(f"[ERROR] No se encontró ningún JSON ('{{') en la respuesta. Respuesta limpia:\n{clean_output}")
            raise ValueError("No JSON object found in Ollama response")

        end_brace_index = self._find_matching_brace(clean_output, start_brace_index)
        if end_brace_index == -1:
             print(f"[ERROR] No se pudo encontrar la llave de cierre '}}'. Respuesta limpia:\n{clean_output}")
             raise ValueError("Malformed JSON, could not find matching closing brace")
        
        json_string = clean_output[start_brace_index : end_brace_index + 1]
        json_string_repaired = re.sub(r'[\n\r\t]', ' ', json_string)
        
        # Limpieza adicional del JSON string
        json_string_repaired = re.sub(r'\s+', ' ', json_string_repaired)  # Elimina espacios múltiples
        json_string_repaired = re.sub(r'(\w+)\s+\1', r'\1', json_string_repaired)  # Elimina palabras repetidas
        json_string_repaired = re.sub(r'(\')\s+\1', r'\1', json_string_repaired)  # Elimina comillas repetidas
        
        print(f"[DEBUG] JSON extraído y limpiado: {json_string_repaired}")

        try:
            # Validación previa del JSON
            if not json_string_repaired.strip().startswith('{') or not json_string_repaired.strip().endswith('}'):
                print("[ERROR] El JSON no comienza con '{' o no termina con '}'")
                raise ValueError("Invalid JSON format")
                
            parsed_json = json.loads(json_string_repaired)
            
            # Si la respuesta tiene una clave 'prompt', la devolvemos directamente
            if 'prompt' in parsed_json:
                print("[+] Respuesta procesada exitosamente")
                return parsed_json
            
            # Si no, intentamos acceder a la clave del nombre de la función
            if function_template['name'] in parsed_json:
                print("[+] Respuesta procesada exitosamente")
                return parsed_json[function_template['name']]
            
            # Si ninguna de las anteriores funciona, devolvemos el JSON completo
            print("[+] Respuesta procesada exitosamente")
            return parsed_json

        except json.JSONDecodeError as e:
            print(f"[ERROR] Falló el parseo de la respuesta JSON de Ollama: {e}")
            print(f"--- JSON String Reparado (intentado) ---\n{json_string_repaired}\n--------------------------")
            raise

    def close(self):
        print("[Ollama Client] Cerrando conexiones SSH...")
        if self.colossus_client:
            self.colossus_client.close()
        if self.gateway_client:
            self.gateway_client.close()
        print("[+] Conexiones cerradas.")

# --- END OF FILE GeneticPromptLab/ollama_ssh_client.py ---