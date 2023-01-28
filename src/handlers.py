from aiogram import types, Dispatcher, Bot

from .config import BOT_TOKEN
from .controllers import HandlersController

bot = Bot(token=BOT_TOKEN)

dp = Dispatcher(bot)


@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    await HandlersController.send_welcome(message)


@dp.message_handler(commands=['help'])
async def send_help(message: types.Message):
    await HandlersController.send_help(message)


@dp.message_handler(content_types=['photo'])
async def handle_photo(message: types.Message):
    await HandlersController.handle_photo(message)
