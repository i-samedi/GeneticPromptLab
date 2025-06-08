import paramiko
import json
import time
import sys
import select
import re # Importamos el módulo de expresiones regulares

# --- Carga de configuración (sin cambios) ---
try:
    with open('ssh_config.json', 'r') as f:
        config = json.load(f)
    gateway_config = config['gateway']
    colossus_config = config['colossus']
    ollama_model = config['ollama_model']
except FileNotFoundError:
    print("Error: El archivo 'ssh_config.json' no se encontró.")
    sys.exit(1)
except KeyError as e:
    print(f"Error: La clave {e} no se encuentra en el archivo de configuración.")
    sys.exit(1)

GATEWAY_HOST = gateway_config['hostname']
GATEWAY_PORT = gateway_config['port']
GATEWAY_USER = gateway_config['username']
GATEWAY_PASS = gateway_config['password']

COLOSSUS_HOST = colossus_config['hostname']
COLOSSUS_USER = colossus_config['username']
COLOSSUS_PASS = colossus_config['password']


# --- NUEVA FUNCIÓN PARA LIMPIAR CÓDIGOS ANSI ---
def clean_ansi_codes(text):
    """
    Elimina las secuencias de escape ANSI (colores, movimiento de cursor, etc.) del texto.
    """
    ansi_escape_pattern = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
    return ansi_escape_pattern.sub('', text)


def read_interactive_output(shell, stop_string=">>> ", timeout=3.0, debug=False):
    """
    Lee la salida de un shell interactivo.
    Añadido un parámetro 'debug' para activar/desactivar los mensajes de depuración.
    """
    output_buffer = ""
    last_data_time = time.time()

    while time.time() - last_data_time < timeout:
        r, _, _ = select.select([shell], [], [], 0.1)
        if r:
            try:
                chunk = shell.recv(4096).decode('utf-8', errors='ignore')
                if not chunk:
                    break
                
                if debug:
                    print(f"[DEBUG] Recibido: {repr(chunk)}")

                output_buffer += chunk
                last_data_time = time.time()

                if stop_string and output_buffer.strip().endswith(stop_string):
                    break
            except Exception as e:
                print(f"\n[ERROR] Error leyendo del shell: {e}")
                break
    
    return output_buffer


if __name__ == '__main__':
    gateway_client = None
    colossus_client = None

    try:
        # --- PASO 1 y 2: Conexión y túnel (sin cambios) ---
        print(f"[*] Conectando al gateway {GATEWAY_HOST}:{GATEWAY_PORT}...")
        gateway_client = paramiko.SSHClient()
        gateway_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        gateway_client.connect(hostname=GATEWAY_HOST, port=GATEWAY_PORT, username=GATEWAY_USER, password=GATEWAY_PASS, timeout=10)
        print("[+] Conexión al gateway establecida.")

        print(f"[*] Abriendo túnel hacia {COLOSSUS_HOST}...")
        transport = gateway_client.get_transport()
        dest_addr = (COLOSSUS_HOST, 22)
        local_addr = ('localhost', 0)
        proxy_channel = transport.open_channel("direct-tcpip", dest_addr, local_addr)
        print("[+] Túnel abierto. Conectando a Colossus...")

        colossus_client = paramiko.SSHClient()
        colossus_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        colossus_client.connect(hostname=COLOSSUS_HOST, username=COLOSSUS_USER, password=COLOSSUS_PASS, sock=proxy_channel)
        print("[+] ¡Conexión a Colossus establecida con éxito!")

        # --- PASO 3 y 4: Shell interactiva con limpieza de salida ---
        print("\n[*] Abriendo shell interactiva en Colossus...")
        shell = colossus_client.invoke_shell(width=120, height=40)

        print("[*] Limpiando banner de bienvenida...")
        read_interactive_output(shell, stop_string=None, timeout=1.5)

        ollama_command = f"ollama run {ollama_model}\n"
        print(f"\n[*] Ejecutando comando: {ollama_command.strip()}")
        shell.send(ollama_command)
        
        print("[*] Esperando que Ollama inicie...")
        read_interactive_output(shell, stop_string=">>> ", timeout=15) # Tiempo para que inicie Ollama
        
        print("\n[+] Ollama está listo. Por favor, introduce tu prompt.")
        print("[i] Escribe '/bye' o 'exit' para terminar la sesión y salir.")

        while True:
            user_prompt = input("\nTu prompt para Mistral: ")
            if user_prompt.lower() in ['/bye', 'exit', 'quit']:
                shell.send("/bye\n")
                time.sleep(1)
                break
            
            shell.send(user_prompt + "\n")
            
            # 1. Leer la salida cruda del shell
            raw_output = read_interactive_output(shell, stop_string=None, timeout=5.0)

            # 2. Limpiar la salida de códigos ANSI y otros artefactos de Ollama
            clean_output = clean_ansi_codes(raw_output)
            
            # 3. Procesar el texto para que sea más legible
            # Ollama añade ">>> Send a message..." al final, lo eliminamos.
            if ">>> Send a message" in clean_output:
                clean_output = clean_output.split(">>> Send a message")[0]
            
            # Eliminamos los spinners de carga que puedan haber quedado
            spinners = ['⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
            for char in spinners:
                clean_output = clean_output.replace(char, '')
            
            # Imprimir la respuesta final y limpia
            print("\nRespuesta de Mistral:")
            print(clean_output.strip())
            
            # Limpiamos el buffer del prompt ">>>" que llega al final
            read_interactive_output(shell, stop_string=">>> ", timeout=1.0)
            
            print("-" * 30)

    except paramiko.AuthenticationException as e:
        print(f"\n[!] Error de autenticación: {e}. Revisa usuario/contraseña.")
    except Exception as e:
        print(f"\n[!] Ocurrió un error: {e}")
    finally:
        print("\n[*] Cerrando conexiones...")
        if colossus_client:
            colossus_client.close()
        if gateway_client:
            gateway_client.close()
        print("[+] Conexiones cerradas. ¡Hasta luego!")