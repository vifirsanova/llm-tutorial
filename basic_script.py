from transformers import AutoTokenizer, Gemma3ForCausalLM, BitsAndBytesConfig
import torch, re

# Конфигурация для 8-битной квантизации модели (уменьшает потребление памяти)
quantization_config = BitsAndBytesConfig(load_in_8bit=True)

# Загрузка предобученной модели Gemma-3-1b-it с квантизацией и токенизатора
model = Gemma3ForCausalLM.from_pretrained('google/gemma-3-1b-it', quantization_config=quantization_config).eval()
tokenizer = AutoTokenizer.from_pretrained('google/gemma-3-1b-it')

# Получение ввода от пользователя
inp = input()

# Формирование структуры сообщения для модели в формате чата
messages = [
    [
        {"role": "system", "content": [{"type": "text", "text": ""}]},  # Системное сообщение (пустое)
        {"role": "user", "content": [{"type": "text", "text": inp}]},   # Пользовательский ввод
    ],
]

# Применение шаблона чата к сообщениям, токенизация и подготовка входных данных для модели
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,  # Добавление промпта для генерации
    tokenize=True,               # Токенизация текста
    return_dict=True,            # Возврат данных в виде словаря
    return_tensors="pt",         # Возврат тензоров PyTorch
).to(model.device)               # Перенос данных на устройство модели (GPU/CPU)

# Генерация ответа модели в режиме инференса (без вычисления градиентов)
with torch.inference_mode():
    outputs = model.generate(
        **inputs,               # Распаковка входных данных
        temperature=0.7,         # Параметр "температуры" для разнообразия ответов
        top_k=50,                # Ограничение на топ-k токенов при генерации
        max_new_tokens=2048,     # Максимальное количество новых токенов в ответе
    )

# Декодирование сгенерированных токенов в текст
outputs = tokenizer.batch_decode(outputs)

# Извлечение ответа модели из текста (между тегами <start_of_turn> и <end_of_turn>)
# и удаление первых 6 символов (лишний префикс)
response = re.findall(r'<start_of_turn>(.*?)<end_of_turn>', outputs[0], re.DOTALL)[1][6:]
print(response)
