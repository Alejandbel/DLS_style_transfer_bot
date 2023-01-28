import asyncio
import os

from aiogram import types

from ..constants import MAX_PHOTO_SIZE, MESSAGE
from src.services import UsersService

users_service = UsersService()


class HandlersController:
    @staticmethod
    async def send_welcome(message: types.Message):
        await message.reply(MESSAGE.WELCOME_MESSAGE.value)

    @staticmethod
    async def handle_photo(message: types.Message):
        user_id = message.from_id
        photo = message.photo[-1]

        if photo.height * photo.width > MAX_PHOTO_SIZE:
            await message.reply(MESSAGE.PHOTO_TOO_LARGE.value)

        users_service.add_photo(user_id, photo)

        if users_service.are_all_photos_of_user_exists(user_id):
            (_, result_root) = await asyncio.gather(message.reply(MESSAGE.PLEASE_WAIT.value),
                                                    users_service.transfer_image_of_user(user_id))
            users_service.clear_or_add_user(user_id)

            with open(result_root, 'rb') as file:
                await message.answer_photo(file, caption=MESSAGE.RESULT.value)

            os.remove(result_root)
        else:
            await message.reply(MESSAGE.FIRST_PICTURE_UPLOADED.value)

    @staticmethod
    async def send_help(message: types.Message):
        await message.answer(MESSAGE.HELP_MESSAGE.value)
