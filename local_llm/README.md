# Чат-бот с потоковой генерацией

В этом проекте представлен пример реализации чат-бота на основе модели Gemma-3-1b-it от Google, которую мы запускаем локально. 

Для запуска этого скрипта вам нужен доступ к графическому процессору GPU и модель Gemma-3-1b-it на вашем устройстве.

## Что можно улучшить в базовом скрипте?

- Потоковая генерация ответов (постепенный вывод токенов)
- Поддержка истории сообщений (для in-context learning)
- 8-битная квантизация для экономии вычислительных ресурсов (добавлена в код)

## Как это добавить?

1. Создать хранилище истории сообщений, которое вы будете подгружать в input модели в пределах контекстного окна либо использовать [готовые инструменты](https://python.langchain.com/docs/concepts/chat_history/)



3. **Историю сообщений**:
   ```python
   messages = [
       {"role": "user", "content": "Привет!"},
       {"role": "assistant", "content": "Здравствуйте! Как я могу помочь?"},
       {"role": "user", "content": "Расскажи о себе"}
   ]
   ```

4. **Семантический поиск**:
   - Использование векторных баз данных (FAISS, Chroma)
   - Поиск по похожим запросам в истории
   - Интеграция через библиотеки `sentence-transformers` и `langchain`

## Схема работы

```mermaid
graph TD
    A[Пользовательский ввод] --> B[Подготовка контекста]
    B --> C[Генерация токенов]
    C --> D[Постепенный вывод]
    D --> E[Обновление истории]
    E --> F[Следующий запрос]
```

## Используемые библиотеки

Основные зависимости:
- `transformers` - работа с моделями HuggingFace
- `torch` - тензорные операции
- `bitsandbytes` - 8-битная квантизация
- `re` - обработка регулярных выражений
- `threading` и `queue` - многопоточность

Дополнительные библиотеки для расширений:
- `sentence-transformers` - эмбеддинги текста
- `faiss` или `chromadb` - векторный поиск
- `langchain` - интеграция компонентов

## Примеры использования

1. Базовый запрос:
   ```python
   python run.py
   > Привет! Кто ты?
   ```

2. С историей диалога (расширенная версия):
   ```python
   from collections import deque
   chat_history = deque(maxlen=5)  # Ограничение длины истории
   ```

3. С семантическим поиском:
   ```python
   from sentence_transformers import SentenceTransformer
   encoder = SentenceTransformer('all-MiniLM-L6-v2')
   ```

## Источники и ссылки

- [Gemma на HuggingFace](https://huggingface.co/google/gemma-3-1b-it)
- [LangChain документация](https://python.langchain.com/)
- [FAISS руководство](https://github.com/facebookresearch/faiss)
- [Sentence Transformers](https://www.sbert.net/)

## Установка

```bash
pip install torch transformers bitsandbytes sentence-transformers faiss-cpu
```
