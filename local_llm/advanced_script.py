from transformers import AutoTokenizer, Gemma3ForCausalLM, BitsAndBytesConfig
import torch, re
from threading import Thread
from queue import Queue

# Инициализация модели и токенизатора с 8-битной квантизацией
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
model = Gemma3ForCausalLM.from_pretrained('google/gemma-3-1b-it', quantization_config=quantization_config).eval()
tokenizer = AutoTokenizer.from_pretrained('google/gemma-3-1b-it')

def generate_response_stream(input_text, output_queue):
    """
    Генерирует ответ модели потоково и помещает части ответа в очередь.
    
    Args:
        input_text (str): Входной текст от пользователя.
        output_queue (Queue): Очередь для передачи сгенерированных частей ответа.
    """
    # Формирование структуры сообщения для модели
    messages = [[{"role": "system",
                "content": [{"type": "text", "text": ""},]},
                {"role": "user",
                "content": [{"type": "text", "text": input_text},]},],]
    
    # Применение шаблона чата и подготовка входных данных
    inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,  # Добавление промпта для генерации
                tokenize=True,               # Токенизация текста
                return_dict=True,            # Возврат данных в виде словаря
                return_tensors="pt",         # Возврат тензоров PyTorch
                ).to(model.device)           # Перенос данных на устройство модели

    # Генерация ответа по токенам
    with torch.inference_mode():
        generated_tokens = []
        for new_token in model.generate(**inputs, 
                                      temperature=0.7,  # Параметр разнообразия ответов
                                      top_k=50,         # Ограничение на топ-k токенов
                                      max_new_tokens=2048,  # Максимальное количество токенов
                                      do_sample=True,   # Включение стохастичности
                                      streamer=None):   # Ручная обработка потока
            
            generated_tokens.append(new_token)
            # Декодирование текущих токенов в текст
            current_output = tokenizer.decode(generated_tokens[0], skip_special_tokens=False)
            
            # Извлечение ответа модели (между тегами <start_of_turn>model и <end_of_turn>)
            matches = re.findall(r'<start_of_turn>model(.*?)(?:<end_of_turn>|$)', current_output, re.DOTALL)
            if matches:
                latest_response = matches[-1].strip()  # Последний найденный ответ
                output_queue.put(latest_response)      # Помещение ответа в очередь

        output_queue.put(None)  # Сигнал завершения генерации

def display_stream(output_queue):
    """
    Отображает потоковый вывод из очереди в реальном времени.
    
    Args:
        output_queue (Queue): Очередь, содержащая части ответа модели.
    """
    current_response = ""
    while True:
        chunk = output_queue.get()  # Получение очередной части ответа
        if chunk is None:  # Признак завершения генерации
            print()  # Переход на новую строку
            break
        
        # Вывод только новой части ответа (чтобы избежать дублирования)
        new_part = chunk[len(current_response):]
        print(new_part, end='', flush=True)  # Вывод без буферизации
        current_response = chunk  # Обновление текущего ответа

# Основной цикл взаимодействия с пользователем
while True:
    inp = input("\nUser: ")  # Получение ввода от пользователя
    if inp.lower() in ['exit', 'quit']:  # Условие выхода
        break
        
    print("\nAssistant: ", end='', flush=True)  # Приглашение для ответа модели
    
    # Создание очереди для обмена данными между потоками
    output_queue = Queue()
    
    # Запуск генерации ответа в отдельном потоке
    gen_thread = Thread(target=generate_response_stream, args=(inp, output_queue))
    gen_thread.start()
    
    # Отображение потокового вывода в основном потоке
    display_stream(output_queue)
    
    # Ожидание завершения потока генерации
    gen_thread.join()
