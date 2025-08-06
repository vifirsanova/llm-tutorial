from transformers import AutoTokenizer, Gemma3ForCausalLM, BitsAndBytesConfig
import torch, re
from threading import Thread
from queue import Queue
from guardrails.hub import ToxicLanguage  # Импорт только ToxicLanguage
from guardrails import Guard

# Инициализация модели и токенизатора с 8-битной квантизацией
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
model = Gemma3ForCausalLM.from_pretrained('google/gemma-3-1b-it', quantization_config=quantization_config).eval()
tokenizer = AutoTokenizer.from_pretrained('google/gemma-3-1b-it')

# Инициализация Guardrail для токсичного языка
toxic_guard = Guard().use(
    ToxicLanguage,
    threshold=0.5,            # Порог определения токсичности (0-1)
    validation_method="sentence",  # Проверка по предложениям
    on_fail="exception"       # Действие при нарушении - исключение
)

def generate_response_stream(input_text, output_queue):
    """
    Генерирует ответ модели потоково с проверкой на токсичность.
    """
    messages = [[{"role": "system",
                "content": [{"type": "text", "text": ""},]},
                {"role": "user",
                "content": [{"type": "text", "text": input_text},]},],]
    
    inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                ).to(model.device)

    with torch.inference_mode():
        generated_tokens = []
        for new_token in model.generate(**inputs, 
                                      temperature=0.7,
                                      top_k=50,
                                      max_new_tokens=2048,
                                      do_sample=True,
                                      streamer=None):
            
            generated_tokens.append(new_token)
            current_output = tokenizer.decode(generated_tokens[0], skip_special_tokens=False)
            
            matches = re.findall(r'<start_of_turn>model(.*?)(?:<end_of_turn>|$)', current_output, re.DOTALL)
            if matches:
                latest_response = matches[-1].strip()
                
                # Проверка на токсичность
                try:
                    toxic_guard.validate(latest_response, metadata={})
                    output_queue.put(latest_response)
                except Exception as e:
                    output_queue.put(f"[ОШИБКА БЕЗОПАСНОСТИ: {str(e)}]")
                    break

        output_queue.put(None)

def display_stream(output_queue):
    current_response = ""
    while True:
        chunk = output_queue.get()
        if chunk is None:
            print()
            break
        
        new_part = chunk[len(current_response):]
        print(new_part, end='', flush=True)
        current_response = chunk

if __name__ == "__main__":   
    while True:
        inp = input("\nUser: ")
        if inp.lower() in ['exit', 'quit']:
            break
            
        print("\nAssistant: ", end='', flush=True)
        output_queue = Queue()
        gen_thread = Thread(target=generate_response_stream, args=(inp, output_queue))
        gen_thread.start()
        display_stream(output_queue)
        gen_thread.join()
