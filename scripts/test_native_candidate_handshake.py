"""Deterministic first-message ordering and the existing real-socket contracts."""
import asyncio
import json
import sys
import unittest

import online_candidate as entry
sys.path[:0] = [str(entry.BASE / 'retry-bundle-operator-source/src'), str(entry.BASE)]
from native_application_control_handshake_v7 import NativeApplicationControlServer


class Peer:
    def __init__(self, fail=False):
        self.messages = []
        self.fail = fail

    async def send(self, message):
        if self.fail:
            raise RuntimeError('fixture hello write failed')
        self.messages.append(json.loads(message))

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration


class State:
    def __init__(self):
        self.phase = 'running'

    def hello(self):
        return {'type': 'control_hello', 'phase': self.phase, 'capability': 'fixture-only'}

    def snapshot(self, *, status=None):
        return {'type': 'control_state', 'phase': self.phase,
                'status': status or self.phase, 'cleanup_confirmed': self.phase == 'cancelled'}


class Handshake(unittest.IsolatedAsyncioTestCase):
    async def test_queued_broadcast_cannot_precede_new_peer_hello(self):
        server = NativeApplicationControlServer('127.0.0.1', 0, State())
        peer = Peer()
        await server._send_lock.acquire()
        publication = asyncio.create_task(server._deliver())
        await asyncio.sleep(0)
        handler = asyncio.create_task(server._handler(peer))
        await asyncio.sleep(0)
        try:
            self.assertNotIn(peer, server._clients)
        finally:
            server._send_lock.release()
            await asyncio.gather(publication, handler)
        self.assertEqual(peer.messages[0]['type'], 'control_hello')

    async def test_failed_hello_does_not_admit_peer_or_claim_delivery(self):
        server = NativeApplicationControlServer('127.0.0.1', 0, State())
        peer = Peer(fail=True)
        with self.assertRaisesRegex(RuntimeError, 'hello write failed'):
            await server._handler(peer)
        self.assertNotIn(peer, server._clients)
        self.assertFalse(server.hello_delivered.is_set())

    async def test_existing_peer_still_receives_queued_authoritative_state(self):
        server = NativeApplicationControlServer('127.0.0.1', 0, State())
        existing, joining = Peer(), Peer()
        server._clients.add(existing)
        await server._send_lock.acquire()
        publication = asyncio.create_task(server._deliver())
        await asyncio.sleep(0)
        handler = asyncio.create_task(server._handler(joining))
        await asyncio.sleep(0)
        server.state.phase = 'cancelled'
        server._send_lock.release()
        await asyncio.gather(publication, handler)
        self.assertEqual(existing.messages[0]['phase'], 'cancelled')
        self.assertEqual(joining.messages[0]['type'], 'control_hello')
        self.assertEqual(joining.messages[0]['phase'], 'cancelled')


def load_tests(loader, tests, pattern):
    return loader.loadTestsFromTestCase(Handshake)


if __name__ == '__main__':
    if '--old' in sys.argv:
        sys.argv.remove('--old')
        from native_application_control_ordered_v6 import NativeApplicationControlServer
    unittest.main()
