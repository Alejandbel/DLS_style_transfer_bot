import asyncio

from aiogram import executor, Dispatcher

from dispatcher_mutators import apply_commands
from handlers import dp


async def on_startup(dp: Dispatcher):
    await apply_commands(dp)


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True, on_startup=on_startup)
