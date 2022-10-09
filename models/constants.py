import sys
sys.path.insert(0, "./models/moretech")
from moretech.classifier import Network

net = Network()
WELCOME = "Здравствуйте! Я буду вашим виртуальным ассистентом в мире новостей и трендов. Вы сможете первым узнавать об изменениях и событиях, которые произошли за последнее время. Но сначала, позвольте мне узнать немного о вас. Расскажите, пожалуйста, кем вы работаете?!"

ROLES = ["Генеральный директор", "Бухгалтер", "Другое"]

MSG_INDUSTRY = "Выберите наиболее релевантные для вас отрасли"

INDUSTRY = net.idx2industry

MSG_TIME = "Введите, в какое время вы хотите получать ежедневный дайджест новостей?\nФормат времени: чч:мм"

MSG_PROFILE_CREATED = "Ваш профиль успешно создан."




