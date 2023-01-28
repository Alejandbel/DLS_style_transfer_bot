from aiogram import Dispatcher, types


async def apply_commands(dp: Dispatcher):
    await dp.bot.set_my_commands([
        types.BotCommand("help", "Помощь"),
        types.BotCommand("start", "Запустить бота"),
    ])
