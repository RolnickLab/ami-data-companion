# gunicorn_conf.py
from multiprocessing import cpu_count

bind = "0.0.0.0:8000"

# Worker Options
workers = cpu_count() + 1
worker_class = 'uvicorn.workers.UvicornWorker'
timeout = 120

# Logging Options
loglevel = 'debug'
accesslog = '/home/debian/logs/access_log'
errorlog =  '/home/debian/logs/error_log'
