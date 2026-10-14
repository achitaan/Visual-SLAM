"""Bounded loopback qualification recorder; no estimator or Stop authority."""
import hashlib
from http.server import BaseHTTPRequestHandler,HTTPServer
import json
import os
from pathlib import Path
import threading
from urllib.parse import urlparse

ORIGIN='http://127.0.0.1:4301'
class ReceiptSink:
    def __init__(self,path,run_id,*,port=0,max_bytes=2*1024*1024):
        self.path=Path(path);self.run_id=run_id;self.max_bytes=max_bytes
        if self.path.exists() or self.path.is_symlink():raise FileExistsError(self.path)
        if type(max_bytes) is not int or max_bytes<=0:raise ValueError('positive request bound required')
        self.count=0;self.error=None;self.digest=hashlib.sha256();self._closed=False
        self.stream=self.path.open('xb');self.token=os.urandom(32).hex();owner=self
        class Handler(BaseHTTPRequestHandler):
            def setup(self):
                super().setup();self.connection.settimeout(2)
            def log_message(self,*args):pass
            def do_POST(self):
                code=200
                try:
                    if self.headers.get('Origin')!=ORIGIN or self.path!='/receipt/'+owner.token:
                        self.send_error(403);return
                    length=int(self.headers.get('Content-Length','0'))
                    if not 0<length<=owner.max_bytes:
                        self.send_error(413);return
                    raw=self.rfile.read(length)
                    if len(raw)!=length:raise ValueError('truncated receipt')
                    item=json.loads(raw)
                    if not isinstance(item,dict) or item.get('run_id')!=owner.run_id:
                        self.send_error(400);return
                    if b'\n' in raw or b'\r' in raw:raise ValueError('one compact JSON record required')
                    if owner.error:raise RuntimeError(owner.error)
                    # Parsing is per request; no cumulative object/string history.
                    encoded=raw+b'\n';owner.stream.write(encoded);owner.stream.flush()
                    owner.digest.update(encoded);owner.count+=1
                except BaseException as exc:
                    owner.error=repr(exc);code=500
                self.send_response(code);self.send_header('Content-Length','0');self.end_headers()
        try:self.server=HTTPServer(('127.0.0.1',port),Handler)
        except BaseException:self.stream.close();raise
        self.url=f'http://127.0.0.1:{self.server.server_port}/receipt/{self.token}'
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True)
    def start(self):self.thread.start()
    def close(self):
        if self._closed:return
        self.server.shutdown();self.server.server_close();self.thread.join(timeout=3)
        self.stream.close();self._closed=True
        if self.thread.is_alive():raise RuntimeError('receipt recorder did not stop')
    def summary(self):
        if not self._closed:raise ValueError('terminal receipt summary required')
        return dict(path=str(self.path),records=self.count,sha256=self.digest.hexdigest(),
                    max_request_bytes=self.max_bytes,error=self.error,thread_stopped=not self.thread.is_alive())
