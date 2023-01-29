# Style transfer bot
## _DLS final project_

This project was created as a final task in [Deep Learning Shool](https://dls.samcs.ru/). Bot deployed on custom linux server on [vultr](https://vultr.com).

## Examle of work

![Screenshot from 2023-01-28 21-56-20](https://user-images.githubusercontent.com/90352952/215285995-1113cb01-f7b3-456d-9f74-81b17ba75580.png)

## Installation

Clone repository then enter it's directory

```sh
cd DLS_style_transfer_bot
```

Create .env file with your bot token

```sh
echo BOT_TOKEN=YOUR_BOT_TOKEN > .env
```
> Note: without .env file it wont even start

Install the dependencies with pip and start the bot.

```sh
pip install -r /requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
python main.py
```

## Docker

Bot is very easy to install and deploy in a Docker container.

By default, the Docker will use .env file, so make sure to create it.
When ready, simply use the Dockerfile to build the image.

```sh
docker build -t <youruser>/style_bot:${version}
```

This will create the bot image and pull in the necessary dependencies.
Be sure to swap out `${version}` with the actual
version of bot.

Once done, run the Docker image.

```sh
docker run -d --restart=always --name=style_bot <youruser>/style_bot:${version}
```

Verify the deployment by simply entering the bot.


Model was taken from [this github repo](https://github.com/zhanghang1989/PyTorch-Multi-Style-Transfer).
