import ast # библиотека для парсинга json

def sampling(INPUT):
  """
  Капсом выделены те переменные, 
  которые вы вводите.

  Используем промпты, которые у вас полуились,
  но их надо будет доработать с учетом того,
  что вам нужно вывести именно транскрипцию
  и записать всё в JSON.
  """
  outputs = client.chat.completions.create(
                             messages=[
                                 # надо изменить системный промпт так, 
                                 # чтобы модель генерировала транскрипцию в формате IPA
                                 # и выводить результат в словарь вида "transcription": "ТРАНСКРИПЦИЯ"
                                 {"role": "system", "content": "ВАШ СИСТЕМНЫЙ ПРОМПТ"
                                 },
                                 {"role": "user",
                                  "content": "ВАШ ПРОМПТ"},
                                 ],
                                 response_format={
                                     "type": "json", # здесь мы задаем формат JSON
                                     "value": {
                                         "properties": {
                                             # ключ оставляем transcription
                                             # тип оставляем string
                                             "transcription": {"type": "string"}, 
                                             }
                                         }
                                     },
                             stream=False,
                             max_tokens=1024, # можно настроить параметры 
                             temperature=0.7, # можно настроить параметры 
                             top_p=0.1        # можно настроить параметры 
                             ).choices[0].get('message')['content']
  outputs = ast.literal_eval(outputs)
  # transcription = outputs['transcription'] # получаем транскприцию (если нужно вытащить ее из словаря, например, для дебага)
  return # функция должна выводить словарь следующего вида:
         # {"sentence": "ТЕКСТ, КОТОРЫЙ ВЫ РАЗМЕЧАЕТЕ", "transcription": "ЕГО ТРАНСКРИПЦИЯ"}
