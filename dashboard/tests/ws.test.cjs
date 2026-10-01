const test = require('node:test');
const assert = require('node:assert/strict');
const { TelemetrySocket } = require('../.test-build/ws.js');

class MockWebSocket {
  static OPEN = 1;
  static CONNECTING = 0;
  static instances = [];
  readyState = 0;
  messages = [];
  constructor(url) { this.url = url; MockWebSocket.instances.push(this); }
  send(data) { this.messages.push(data); }
  close() { this.readyState = 3; this.onclose?.(); }
}
global.WebSocket = MockWebSocket;

test('connection state follows events and controls require an open socket', () => {
  const socket = new TelemetrySocket('ws://localhost');
  const states = [];
  socket.onStatus((state) => states.push(state));
  socket.connect();
  const ws = MockWebSocket.instances.at(-1);
  assert.deepEqual(states, []);
  socket.sendControl({ type: 'control', action: 'stop' });
  assert.equal(ws.messages.length, 0);
  ws.readyState = MockWebSocket.OPEN;
  ws.onopen();
  assert.deepEqual(states, [true]);
  socket.sendControl({ type: 'control', action: 'stop' });
  assert.equal(JSON.parse(ws.messages[0]).action, 'stop');
  socket.close();
  assert.equal(states.at(-1), false);
});

test('malformed and unrelated messages do not reach frame listeners', () => {
  const socket = new TelemetrySocket('ws://localhost');
  const frames = [];
  socket.onFrame((frame) => frames.push(frame));
  socket.connect();
  const ws = MockWebSocket.instances.at(-1);
  for (const data of ['invalid', 'null', '[]', '{}']) ws.onmessage({ data });
  assert.equal(frames.length, 0);
  ws.onmessage({ data: JSON.stringify({ schema_version: 1, frame_index: 1, pose_T_wc: [] }) });
  assert.equal(frames.length, 1);
  socket.close();
});

test('disconnect reconnects and explicit close cancels retries', () => {
  const originalSet = global.setTimeout;
  const originalClear = global.clearTimeout;
  let pending;
  global.setTimeout = (callback) => { pending = callback; return 1; };
  global.clearTimeout = () => { pending = undefined; };
  try {
    const socket = new TelemetrySocket('ws://localhost');
    socket.connect();
    const first = MockWebSocket.instances.at(-1);
    first.close();
    assert.equal(typeof pending, 'function');
    pending();
    assert.notEqual(MockWebSocket.instances.at(-1), first);
    socket.close();
    assert.equal(pending, undefined);
    const count = MockWebSocket.instances.length;
    socket.connect();
    assert.equal(MockWebSocket.instances.length, count);
  } finally {
    global.setTimeout = originalSet;
    global.clearTimeout = originalClear;
  }
});
