"""Bounded ordered prototype delivery; never discard an original-frame message.

This is isolated from production telemetry. A successful publish means the
owned sender completed for current clients before its absolute deadline. It
does not assert browser receipt: that requires the captured live verifier.
"""
import asyncio
from dataclasses import dataclass,field
import json,math,queue,threading,time
from telemetry import TelemetryServer

@dataclass
class _Delivery:
    text:str
    value:dict
    deadline:float
    done:threading.Event=field(default_factory=threading.Event)
    cancelled:bool=False
    error:object=None

class NativeFifoTelemetryServer(TelemetryServer):
    def __init__(self,*args,publish_timeout_s=2.0,send_timeout_s=.5,max_message_bytes=1024*1024,**kwargs):
        if not math.isfinite(publish_timeout_s) or not .1<=publish_timeout_s<=5:
            raise ValueError('bounded publish deadline required')
        if not math.isfinite(send_timeout_s) or not .01<=send_timeout_s<publish_timeout_s:
            raise ValueError('send deadline below publish deadline required')
        if type(max_message_bytes) is not int or not 1024<=max_message_bytes<=4*1024*1024:
            raise ValueError('bounded message capacity required')
        super().__init__(*args,**kwargs)
        assert self._queue.maxsize==2
        self.publish_timeout_s=publish_timeout_s;self.send_timeout_s=send_timeout_s;self.max_message_bytes=max_message_bytes
        self._producer=threading.Lock();self._accepted=0;self._delivered=0;self._failed=0;self._peak=0

    @property
    def delivery_statistics(self):
        return dict(accepted_messages=self._accepted,delivered_messages=self._delivered,failed_publications=self._failed,
                    peak_queue=self._peak,queue_capacity=2,publish_timeout_s=self.publish_timeout_s,
                    send_timeout_s=self.send_timeout_s,recipient_receipt_requires_browser_proof=True)

    def publish(self,payload):
        text=json.dumps(payload,allow_nan=False)
        if len(text.encode('utf-8'))>self.max_message_bytes:raise ValueError('native display message exceeds byte cap')
        request=_Delivery(text,json.loads(text),time.monotonic()+self.publish_timeout_s)
        # One ordered publication owner; serialized ownership is also bounded.
        if not self._producer.acquire(timeout=self.publish_timeout_s):raise TimeoutError('publication owner deadline')
        try:
            if self._thread is None or not self._thread.is_alive() or self._stopped.is_set() or self._error:
                raise RuntimeError('native telemetry worker unavailable')
            remaining=request.deadline-time.monotonic()
            if remaining<=0:raise TimeoutError('publication admission deadline')
            try:self._queue.put(request,timeout=remaining)
            except queue.Full:raise TimeoutError('bounded native queue admission deadline') from None
            self._accepted+=1;self._peak=max(self._peak,self._queue.qsize())
            remaining=request.deadline-time.monotonic()
            if remaining<=0 or not request.done.wait(remaining):
                request.cancelled=True;raise TimeoutError('native sender deadline; message not silently dropped')
            if request.error is not None:raise RuntimeError('native delivery incomplete') from request.error
        except BaseException:
            request.cancelled=True;self._failed+=1;raise
        finally:self._producer.release()

    async def _broadcast_loop(self):
        current=None
        try:
            while not self._stopped.is_set():
                if not self._clients:
                    # Keep pending data; the producer deadline bounds missing clients.
                    await asyncio.sleep(.005);continue
                try:current=self._queue.get_nowait()
                except queue.Empty:await asyncio.sleep(.005);continue
                if current.cancelled or time.monotonic()>=current.deadline:
                    current.error=TimeoutError('cancelled native delivery');current.done.set();current=None;continue
                if not self.state.stream_enabled:raise RuntimeError('display stream paused; ordered delivery incomplete')
                timeout=min(self.send_timeout_s,current.deadline-time.monotonic())
                if timeout<=0:raise TimeoutError('native send admission deadline')
                results=await asyncio.wait_for(asyncio.gather(*(client.send(current.text) for client in list(self._clients)),
                                                             return_exceptions=True),timeout)
                if any(isinstance(result,BaseException) for result in results):raise RuntimeError('native client send failed')
                if current.cancelled or time.monotonic()>=current.deadline:raise TimeoutError('late send cannot qualify')
                self._latest=current.value;self._delivered+=1;current.done.set();current=None
        except BaseException as error:
            if current is not None:current.error=error;current.done.set()
            raise
        finally:
            while True:
                try:pending=self._queue.get_nowait()
                except queue.Empty:break
                pending.error=RuntimeError('native sender stopped');pending.done.set()
