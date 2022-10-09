## Описание проекта

Cозданный в рамках хакатона продукт позволяет выявлять тренды и инсайты, сообщать о них клиентам ВТБ Банка с помощью телеграм бота. 
На данный момент новости формируются по двум специализациям и обобщению, но в будущем число может быть увеличено. 
Кроме этого, каждая новость обладает дополнительной характеристикой — отраслью, к которой она принадлежит. 
Было проанализировано 10296 новостей из 6 ключевых источников.
Для удобного добавления новых источников был реализован веб-сервис, а также универсальный парсер, позволяющий получать самые свежие новости.

## Необходимые библиотеки python
pandas
numpy
sklearn
torch
transformers
tqdm
pickle
nltk
pymorphy2
matplotlib
pyTelegramBotApi

## Запуск проекта

телеграм-бот: https://t.me/vtb_trends_bot

1. /start - запускает бота
2. Выбор роли: генеральный директор/бухгалтер/другое
3. Выбор интересующих пользователя отраслей
4. Выбор времени для автоматической рассылки свежих новостей
5. По нажатию кнопки "Дайджест" можно досрочно получить выжимку из актуальных новостей

На текущий момент в чат боте доступен только функционал вывода ежедневных новостей для того, чтобы протестировать формат взаимодействия с клиентами.
В дальнейшем планируется расширение функционала до оповещения о трендах и инсайтах.
Тем не менее в данный момент есть возможность получать тренды и инсайты непосредственно из API.


веб-сервис: http://89.108.88.166:3000/

1. Для входа используйте Логин/Пароль: admin/admin
2. Сервис позволяет добавлять дополнительные новостные ресурсы