from transformers import AutoTokenizer, Gemma3ForCausalLM, BitsAndBytesConfig
import torch, re
from threading import Thread
from queue import Queue
from smolagents import CodeAgent, WebSearchTool, InferenceClientModel

# Инициализация модели и токенизатора с 8-битной квантизацией
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
model = Gemma3ForCausalLM.from_pretrained('google/gemma-3-1b-it', quantization_config=quantization_config).eval()
tokenizer = AutoTokenizer.from_pretrained('google/gemma-3-1b-it')

# Инициализация агента
agent_model = InferenceClientModel()
agent = CodeAgent(tools=[WebSearchTool()], model=agent_model, stream_outputs=True)

def generate_response_stream(input_text, output_queue):
    """
    Генерирует ответ модели потоково и помещает части ответа в очередь.
    
    Args:
        input_text (str): Входной текст от пользователя.
        output_queue (Queue): Очередь для передачи сгенерированных частей ответа.
    """
    # Проверяем, нужно ли использовать агента (по ключевым словам)
    use_agent = any(keyword in input_text.lower() for keyword in ['код', 'программу', 'найди', 'посчитай'])
    
    if use_agent:
        # Используем агента для сложных запросов:
        output_queue.put("\n[Использую агента для обработки запроса...]\n")
        
        # Запускаем агента и получаем шаги выполнения
        steps = []
        def step_callback(step):
            steps.append(step)
            output_queue.put(f"\n[Шаг агента]: {step}\n")
        
        result = agent.run(input_text)
        output_queue.put(f"\n[Результат агента]: {result}\n")
    else:
        # Используем обычную генерацию для простых запросов
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

        # Генерация ответа по токенам
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
                    output_queue.put(latest_response)

    output_queue.put(None)  # Сигнал завершения генерации

def display_stream(output_queue):
    """
    Отображает потоковый вывод из очереди в реальном времени.
    
    Args:
        output_queue (Queue): Очередь, содержащая части ответа модели.
    """
    current_response = ""
    while True:
        chunk = output_queue.get()
        if chunk is None:
            print()
            break
        
        new_part = chunk[len(current_response):]
        print(new_part, end='', flush=True)
        current_response = chunk

# Основной цикл взаимодействия с пользователем
print("Система готова к работе. Введите ваш запрос (или 'exit' для выхода):")
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
