import logging

from dotenv import dotenv_values

logging.basicConfig()

_config = dotenv_values()

BOT_TOKEN = _config['BOT_TOKEN']
