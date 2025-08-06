## Ставим ограничения на модель

- Устанавливаем `guardrails-ai`
- Прописываем свой ключ через `guardrails configure` в терминале (получаем здесь: https://hub.guardrailsai.com/keys)
- Создаем виртуальное окружение
  ```
  python3 -m venv .venv
  source .venv/bin/activate
  ```
- Ищем в хабе нужный рейл, например, https://hub.guardrailsai.com/validator/guardrails/toxic_language
- Устанавливаем рейл по инструкции
- Устанавливаем недостающие компоненты (например, `nltk`)
- Добавляем правило в скрипт
- Запускаем модель и проверяем несколько промптов, нарушающих правило
