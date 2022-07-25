from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
from aiogram.utils import executor
import os
# import tensorflow 

from config import TOKEN
from Fake_news_BERT import check_fake
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

bot = Bot(token=TOKEN)
dp = Dispatcher(bot)

@dp.message_handler(commands=['start'])
async def process_start_command(message: types.Message):
    await message.reply("Привет!\nПришли мне новость и я скажу, настоящая она или фейк")


@dp.message_handler(commands=['help'])
async def process_help_command(message: types.Message):
    await message.reply("Напиши мне что-нибудь, и я отправлю этот текст тебе в ответ!")


@dp.message_handler()
async def echo_message(msg: types.Message):
    prompt = msg.text
    ans = check_fake(prompt)
    await bot.send_message(msg.from_user.id, ans)

@dp.message_handler(content_types=['photo'])
async def get_photo(message: types.Message):
    user_id = message.from_user.id
    img_path = 'input.jpg'
    await message.photo[-1].download(img_path)
    radio_generate()
    await bot.send_photo(user_id, photo=open('result.png', 'rb'))


if __name__ == '__main__':
    executor.start_polling(dp)