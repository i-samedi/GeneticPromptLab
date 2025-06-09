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
        # ... (El constructor __init__ no cambia, puedes dejarlo como está)
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
        """Encuentra el índice de la llave de cierre '}' que corresponde a una de apertura '{'."""
        if text[start_index] != '{':
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
        
        properties = function_template.get("parameters", {}).get("properties", {})
        json_structure_guide = json.dumps({function_template['name']: properties})

        # --- INGENIERÍA DE PROMPTS ULTRA-EXPLÍCITA ---
        full_prompt = (
            f"You are a robot that only responds with valid JSON. Your entire response must be a single JSON object and nothing else. "
            f"The user wants this: '{user_prompt}'. "
            f"The system instructions are: '{system_prompt}'. "
            f"Generate a JSON object that strictly follows this exact structure: {json_structure_guide}. "
            f"IMPORTANT: All string values inside the JSON must be on a single line. Do not use newline characters inside strings. "
            f"Do not include any text, explanations, or markdown formatting like ```json. "
            f"Your entire response must begin with '{{' and end with '}}'. Start now:\n"
        )
        
        full_prompt = full_prompt.replace('\n', ' ').strip() + '\n'

        self.shell.send(full_prompt)
        raw_output = self._read_interactive_output(timeout=30.0)

        # --- LÓGICA DE PARSEO DE "LIMPIEZA Y REPARACIÓN" ---
        clean_output = self._clean_ansi_codes(raw_output)
        
        start_brace_index = clean_output.rfind('{')
        if start_brace_index == -1:
            print(f"[ERROR] No se encontró ningún JSON ('{{') en la respuesta de Ollama. Respuesta limpia:\n{clean_output}")
            raise ValueError("No JSON object found in Ollama response")

        end_brace_index = self._find_matching_brace(clean_output, start_brace_index)
        if end_brace_index == -1:
             print(f"[ERROR] No se pudo encontrar la llave de cierre '}}' para el JSON. Respuesta limpia:\n{clean_output}")
             raise ValueError("Malformed JSON, could not find matching closing brace")
        
        # 1. Extraer el bloque JSON de forma precisa
        json_string = clean_output[start_brace_index : end_brace_index + 1]
        
        # 2. "Reparar" el problema de los saltos de línea ilegales
        json_string_repaired = json_string.replace('\n', ' ').replace('\r', '')

        try:
            parsed_json = json.loads(json_string_repaired)
            return parsed_json[function_template['name']]
        except (json.JSONDecodeError, KeyError) as e:
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