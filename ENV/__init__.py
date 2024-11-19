import logging
LOGGER = logging.getLogger("GLOBAL")
handler = logging.FileHandler("logging.log", mode="w")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

handler.setFormatter(formatter)
LOGGER.addHandler(handler)
LOGGER.setLevel(logging.DEBUG)
