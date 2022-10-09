from datetime import datetime, time
import telebot
from telebot import types
from  constants import WELCOME, ROLES, MSG_INDUSTRY, INDUSTRY, MSG_TIME, MSG_PROFILE_CREATED 

# import sys
# sys.path.insert(0, "./models/moretech")
from moretech.classifier import Network


BOT_TOKEN = '5686816274:AAFt9rm2Dyaze-ciyn1mJO_69tZ0TWf5cj8'
bot = telebot.TeleBot(BOT_TOKEN)

network = Network()
user_dict = {}


class User:
    def __init__(self, role):
        self.role = role
        self.time = '17:00'
        self.industry = []


def interval_job(id):
    bot.send_message(id, "it's your daily digest")
    

@bot.message_handler(commands=['start'])
def send_welcome(message):
    # msg = bot.reply_to(message, WELCOME)
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
    buttons = []
    for role in ROLES:
        buttons.append(types.KeyboardButton(role))
    markup.add(*buttons)
    msg = bot.send_message(message.chat.id, WELCOME, reply_markup=markup)
    bot.register_next_step_handler(msg, process_role_step)


def process_role_step(message):
    chat_id = message.chat.id
    role = message.text
    user = User(role)
    user_dict[chat_id] = user
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
    buttons = []
    for role in INDUSTRY.values():
        buttons.append(types.KeyboardButton(role))
    markup.add(*buttons)
    msg = bot.send_message(message.chat.id, MSG_INDUSTRY, reply_markup=markup)
    bot.register_next_step_handler(msg, process_industry_step)


def process_industry_step(message):
    chat_id = message.chat.id
    industry = message.text
    user_dict[chat_id].industry.append(industry)
    
    markup = types.ReplyKeyboardMarkup(one_time_keyboard=True, resize_keyboard=True)
    button1 = types.KeyboardButton('Да')
    button2 = types.KeyboardButton('Нет')
    markup.add(button1, button2)
    msg = bot.send_message(message.chat.id, 'Добавить еще отрасль?', reply_markup=markup)
    bot.register_next_step_handler(msg, add_industry_step)


def add_industry_step(message):
    if message.text == "Да":
        markup = types.ReplyKeyboardMarkup(one_time_keyboard=True, resize_keyboard=True)
        buttons = []
        for role in INDUSTRY.values():
            buttons.append(types.KeyboardButton(role))
        markup.add(*buttons)
        msg = bot.send_message(message.chat.id, 'Выберите отрасль:', reply_markup=markup)
        bot.register_next_step_handler(msg, process_industry_step)
    elif message.text == "Нет":
        msg = bot.send_message(message.chat.id, MSG_TIME)
        bot.register_next_step_handler(msg, process_time_step)
        

def process_time_step(message):
    try:
        chat_id = message.chat.id
        inp = message.text.split(':')
        user_dict[chat_id].time = inp
        # bot.send_message(MSG_PROFILE_CREATED)
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        button1 = types.KeyboardButton('Дайджест')
        markup.add(button1)
        msg = bot.send_message(message.chat.id, MSG_PROFILE_CREATED, reply_markup=markup)
    except Exception as e:
        bot.reply_to(message, 'Упс, неверный формат времени')


@bot.message_handler(content_types=['text'])
def send_news(message):
    if message.text == "Дайджест":

        msg = bot.send_message(message.chat.id, *network.bot_api(user_dict[message.chat.id].role, user_dict[message.chat.id].industry))
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        button1 = types.KeyboardButton('Дайджест')
        markup.add(button1)
    

bot.enable_save_next_step_handlers(delay=2)
bot.load_next_step_handlers()

# bot.infinity_polling()
def main_loop():
    bot.polling(True)
    while 1:
        for id, user in user_dict.items():
            cur = datetime.now().strftime("%H:%M")
            user_time = user.time
            if cur == user_time:
                interval_job(id)
        
if __name__ == '__main__':
    main_loop()
