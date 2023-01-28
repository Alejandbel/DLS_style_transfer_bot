from aiogram.types import PhotoSize

from .image_transfer_service import ImageTransferService


class User:
    photo_to_transfer: PhotoSize | None
    photo_to_get_style: PhotoSize | None

    def __init__(self):
        self.photo_to_transfer = None
        self.photo_to_get_style = None

    def __int__(self, photo_to_transfer: PhotoSize, photo_to_get_style: PhotoSize):
        self.photo_to_transfer = photo_to_transfer
        self.photo_to_get_style = photo_to_get_style

    def add_photo_if_not_exists(self, photo: PhotoSize) -> None:
        if self.photo_to_transfer is None:
            self.photo_to_transfer = photo
        elif self.photo_to_get_style is None:
            self.photo_to_get_style = photo

    def are_all_photos_exists(self) -> bool:
        return not (self.photo_to_get_style is None or self.photo_to_transfer is None)


class UsersService:
    users: dict[int, User]

    def __init__(self):
        self.users = dict()

    def clear_or_add_user(self, user_id) -> User:
        user = User()
        self.users[user_id] = user
        return user

    def add_photo(self, user_id: int, photo: PhotoSize):
        user = self.users.get(user_id, None)

        if user is None:
            user = self.clear_or_add_user(user_id)

        user.add_photo_if_not_exists(photo)

    def are_all_photos_of_user_exists(self, user_id: int):
        return self.users[user_id].are_all_photos_exists()

    async def transfer_image_of_user(self, user_id):
        user = self.users[user_id]
        return await ImageTransferService.transfer(user.photo_to_transfer, user.photo_to_get_style)
