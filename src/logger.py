import logging
import sys

from tqdm.auto import tqdm


class TqdmHandler(logging.StreamHandler):
	def emit(self, record):
		try:
			msg = self.format(record)
			tqdm.write(msg)
		except Exception:
			self.handleError(record)


def get_logger(name: str = "patent_project") -> logging.Logger:
	logger = logging.getLogger(name)
	if logger.handlers:
		return logger
	logger.setLevel(logging.INFO)
	handler = TqdmHandler(stream=sys.stdout)
	formatter = logging.Formatter(
		"[%(asctime)s] %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
	)
	handler.setFormatter(formatter)
	logger.addHandler(handler)
	return logger
