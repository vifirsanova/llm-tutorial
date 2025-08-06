from transformers import AutoTokenizer, Gemma3ForCausalLM, BitsAndBytesConfig
import torch, re
from threading import Thread
from queue import Queue

# Инициализация модели и токенизатора с 8-битной квантизацией
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
model = Gemma3ForCausalLM.from_pretrained('google/gemma-3-1b-it', quantization_config=quantization_config).eval()
tokenizer = AutoTokenizer.from_pretrained('google/gemma-3-1b-it')

# Системные промпты
SYSTEM_PROMPTS = {
    'markdown': """
Ты - опытный аналитик данных, специализирующийся на работе с таблицами. 
Форматируй ответы в Markdown (*.md) с четкой структурой:

1. Заголовки разделов с ##
2. Таблицы в формате:
   | Столбец 1 | Столбец 2 |
   |-----------|-----------|
   | Данные    | Данные    |
3. Код в блоках ``` с указанием языка
4. Списки с нумерацией или маркерами
""",
    
    'json': """
Ты - эксперт по обработке структурированных данных. Форматируй ответы в JSON (*.json) формате:
{
  "description": "краткое описание задачи",
  "steps": ["шаг 1", "шаг 2"],
  "tables": [
    {
      "name": "название таблицы",
      "columns": ["колонка1", "колонка2"],
      "data": [["значение1", "значение2"]]
    }
  ],
  "code": {
    "language": "python",
    "content": "код для решения"
  }
}
"""
}

def generate_response_stream(input_text, output_queue, prompt_type='markdown'):
    """
    Генерирует ответ модели потоково с выбранным системным промптом.
    
    Args:
        input_text (str): Входной текст от пользователя.
        output_queue (Queue): Очередь для передачи сгенерированных частей ответа.
        prompt_type (str): Тип системного промпта ('markdown' или 'json').
    """
    system_prompt = SYSTEM_PROMPTS.get(prompt_type, SYSTEM_PROMPTS['markdown'])
    
    # Формирование структуры сообщения для модели
    messages = [[{"role": "system",
                "content": [{"type": "text", "text": system_prompt},]},
                {"role": "user",
                "content": [{"type": "text", "text": input_text},]},],]
    
    # Применение шаблона чата
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
while True:
    print("\nДоступные форматы ответов:")
    print("1. Markdown (md)")
    print("2. JSON (json)")
    format_choice = input("Выберите формат ответа (1/2): ").strip().lower()
    prompt_type = 'markdown' if format_choice in ('1', 'md') else 'json'
    
    print(f"\nВыбран формат: {prompt_type.upper()}")
    print("Системный промпт:")
    print(SYSTEM_PROMPTS[prompt_type])
    
    inp = input("\nUser: ")
    if inp.lower() in ['exit', 'quit']:
        break
        
    print("\nAssistant: ", end='', flush=True)
    output_queue = Queue()
    
    gen_thread = Thread(target=generate_response_stream, args=(inp, output_queue, prompt_type))
    gen_thread.start()
    display_stream(output_queue)
    gen_thread.join()
