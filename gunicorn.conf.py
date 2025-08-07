
bind = "0.0.0.0:5000"
workers = 1
worker_class = "sync"
timeout = 60
graceful_timeout = 30
keepalive = 2
max_requests = 1000
max_requests_jitter = 100
preload_app = True
