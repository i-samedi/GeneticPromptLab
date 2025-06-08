import paramiko
import json
import time
import sys
import select

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

# --- NUEVA FUNCIÓN DE LECTURA INTERACTIVA ---
def read_interactive_output(shell, stop_string=">>> ", timeout=3.0):
    """
    Lee la salida de un shell interactivo de forma más robusta.
    Espera datos hasta que no llega nada nuevo durante 'timeout' segundos,
    o hasta que encuentra el 'stop_string'.
    """
    output = ""
    start_time = time.time()
    last_data_time = start_time

    while time.time() - last_data_time < timeout:
        # select.select nos permite esperar datos sin bloquear el programa
        # Esperamos como máximo 0.1 segundos antes de volver a comprobar el timeout
        r, _, _ = select.select([shell], [], [], 0.1)
        if r:
            # Hay datos disponibles para leer
            try:
                chunk = shell.recv(4096).decode('utf-8', errors='ignore')
                if not chunk:
                    # El canal se cerró
                    print("\n[DEBUG] El canal SSH se ha cerrado.")
                    break
                
                # Imprimimos los datos crudos para depuración
                # repr() muestra caracteres especiales como '\n', '\r', etc.
                print(f"[DEBUG] Recibido: {repr(chunk)}")

                # Imprimimos la salida para el usuario en tiempo real
                print(chunk, end='', flush=True)

                output += chunk
                last_data_time = time.time() # Reiniciamos el temporizador de inactividad

                # Si encontramos la cadena de parada, terminamos
                if stop_string and output.strip().endswith(stop_string):
                    break

            except Exception as e:
                print(f"\n[ERROR] Error leyendo del shell: {e}")
                break
        
        # Si no hay datos, el bucle continuará hasta que se agote el timeout
    
    return output


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

        # --- PASO 3 y 4: Shell interactiva (lógica mejorada) ---
        print("\n[*] Abriendo shell interactiva en Colossus...")
        shell = colossus_client.invoke_shell(width=120, height=40) # Damos un tamaño a la terminal virtual

        # Limpiamos el mensaje de bienvenida del servidor
        print("[*] Limpiando banner de bienvenida...")
        read_interactive_output(shell, stop_string=None, timeout=1.5)

        # Ejecutar el comando de ollama
        ollama_command = f"ollama run {ollama_model}\n"
        print(f"\n[*] Ejecutando comando: {ollama_command.strip()}")
        shell.send(ollama_command)

        # Esperamos a que ollama inicie y muestre su prompt ">>>"
        print("[*] Esperando que Ollama inicie...")
        read_interactive_output(shell, stop_string=">>> ", timeout=15) # Más tiempo por si descarga el modelo
        
        print("\n[+] Ollama está listo. Por favor, introduce tu prompt.")
        print("[i] Escribe '/bye' o 'exit' para terminar la sesión con Ollama y salir.")

        while True:
            user_prompt = input("\nTu prompt para Mistral: ")
            if user_prompt.lower() in ['/bye', 'exit', 'quit']:
                shell.send("/bye\n")
                time.sleep(1)
                break
            
            # Enviamos el prompt del usuario a la shell
            shell.send(user_prompt + "\n")
            
            # --- LÓGICA DE LECTURA MEJORADA ---
            # 1. Leemos la respuesta. La función esperará hasta que no haya más texto
            #    llegando durante 'timeout' segundos. No buscamos ">>>" aquí.
            print("\nRespuesta de Mistral:")
            response = read_interactive_output(shell, stop_string=None, timeout=5.0)
            
            # 2. Después de obtener la respuesta, es probable que el prompt ">>>"
            #    ya esté en el buffer. Hacemos una lectura rápida final para limpiarlo.
            read_interactive_output(shell, stop_string=">>> ", timeout=1.0)
            
            print("-" * 30)
            # El prompt para el usuario se vuelve a imprimir por el bucle `input()`

        print("\n[*] Sesión con Ollama terminada.")

    except paramiko.AuthenticationException as e:
        print(f"\n[!] Error de autenticación: {e}. Revisa usuario/contraseña.")
    except Exception as e:
        print(f"\n[!] Ocurrió un error: {e}")
    finally:
        print("[*] Cerrando conexiones...")
        if colossus_client:
            colossus_client.close()
        if gateway_client:
            gateway_client.close()
        print("[+] Conexiones cerradas. ¡Hasta luego!")