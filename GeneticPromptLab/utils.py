import json
import time

def send_query2gpt(client, messages, function_template, temperature=0, pause=5):
    """
    Función adaptada para trabajar con el nuevo OllamaSSHClient.

    Args:
        client: Una instancia de OllamaSSHClient.
        messages (list): Una lista de diccionarios de mensajes (role, content).
        function_template (dict): El diccionario que describe la estructura JSON esperada.
        temperature (int, optional): No se utiliza con Ollama, se mantiene por compatibilidad. Defaults to 0.
        pause (int, optional): Segundos de espera después de la consulta. Defaults to 5.

    Returns:
        dict: La respuesta JSON parseada desde Ollama, emulando la salida de OpenAI.
    """
    try:
        generated_response = client.run_prompt_and_get_json(messages, function_template)
        time.sleep(pause) # Mantenemos la pausa para no sobrecargar el modelo
        return generated_response
    except Exception as e:
        print(f"Error en send_query2gpt al llamar al cliente Ollama: {e}")
        raise