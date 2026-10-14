"""Bounded standard-JSON observation reader; never materialize telemetry history."""
import json
import math

def unique_object(pairs):
    result={}
    for key,value in pairs:
        if key in result:raise ValueError('duplicate observation object key')
        result[key]=value
    return result

def nonfinite(value):raise ValueError('nonfinite observation value: '+value)

def finite_float(token):
    value=float(token)
    if not math.isfinite(value):raise ValueError('nonfinite observation numeric value')
    return value

class ObservationStream:
    def __init__(self,stream,*,chunk_chars=65536,max_item_chars=4*1024*1024):
        if type(chunk_chars) is not int or type(max_item_chars) is not int or not 0<chunk_chars<=max_item_chars:
            raise ValueError('positive bounded chunk/item sizes required')
        self.stream=stream;self.chunk=chunk_chars;self.limit=max_item_chars
        self.buffer='';self.eof=False;self.max_buffer_chars=0
        self.decoder=json.JSONDecoder(object_pairs_hook=unique_object,parse_constant=nonfinite,parse_float=finite_float)

    def fill(self):
        if len(self.buffer)>=self.limit:raise ValueError('observation item exceeds declared bound')
        text=self.stream.read(min(self.chunk,self.limit-len(self.buffer)))
        if not text:self.eof=True;return
        self.buffer+=text;self.max_buffer_chars=max(self.max_buffer_chars,len(self.buffer))

    def whitespace(self):
        while True:
            self.buffer=self.buffer.lstrip()
            if self.buffer or self.eof:return
            self.fill()

    def peek(self):
        self.whitespace()
        if not self.buffer:raise ValueError('truncated observation')
        return self.buffer[0]

    def consume(self,expected):
        if self.peek()!=expected:raise ValueError('observation delimiter differs: '+expected)
        self.buffer=self.buffer[1:]

    def value(self):
        self.whitespace()
        while True:
            try:
                value,end=self.decoder.raw_decode(self.buffer)
                # raw_decode may accept a numeric prefix at a chunk boundary.
                # Wait for its delimiter before committing the scalar value.
                if type(value) in (int,float) and (end==len(self.buffer) or self.buffer[end] not in ' \t\r\n,]}'):
                    if self.eof:raise ValueError('invalid/truncated numeric observation')
                    self.fill();continue
                self.buffer=self.buffer[end:];return value
            except json.JSONDecodeError as exc:
                if self.eof:raise ValueError('invalid/truncated observation item') from exc
                self.fill()

    def events(self):
        self.consume('{');seen=set();telemetry=False
        if self.peek()=='}':raise ValueError('telemetry field required')
        while True:
            key=self.value()
            if not isinstance(key,str) or key in seen:raise ValueError('unique string top-level keys required')
            seen.add(key);self.consume(':')
            if key=='telemetry':
                telemetry=True;self.consume('[')
                if self.peek()!=']':
                    while True:
                        message=self.value()
                        if not isinstance(message,dict):raise ValueError('telemetry object required')
                        yield ('message',message)
                        if self.peek()==']':break
                        self.consume(',')
                self.consume(']')
            else:yield ('field',(key,self.value()))
            if self.peek()=='}':break
            self.consume(',')
        self.consume('}');self.whitespace()
        while not self.eof:
            if self.buffer.strip():raise ValueError('trailing observation content')
            self.fill();self.whitespace()
        if self.buffer.strip() or not telemetry:raise ValueError('invalid final observation coverage')
