import asyncio
import json
import numpy as np
import pytest
import websockets
from telemetry import TelemetryServer, TelemetryState, make_frame_message


def payload(index=1):
    return make_frame_message(frame_index=index, timestamp=1., pose_T_wc=np.eye(4),
                              tracking={}, map_state={'keyframes': 0, 'map_points': 0})


def test_invalid_controls_and_bounded_queue():
    server = TelemetryServer('127.0.0.1', 0)
    for message in ('not json', '[]', 'null', '1'):
        server._handle_message(message)
    for i in range(100):
        server.publish(payload(i))
    assert server._queue.qsize() <= 2
    assert server._latest['frame_index'] == 99


def test_websocket_late_join_pause_resume_and_shutdown():
    server = TelemetryServer('127.0.0.1', 0)
    server.start()
    server.publish(payload())

    async def client():
        async with websockets.connect(f'ws://127.0.0.1:{server.port}') as socket:
            frame = json.loads(await asyncio.wait_for(socket.recv(), 2))
            assert frame['frame_index'] == 1
            await socket.send(json.dumps({'type': 'control', 'action': 'stop'}))
            while True:
                frame = json.loads(await asyncio.wait_for(socket.recv(), 2))
                if not frame['stream_enabled']:
                    break
            await socket.send(json.dumps({'type': 'control', 'action': 'start'}))
            assert json.loads(await asyncio.wait_for(socket.recv(), 2))['stream_enabled']
    try:
        asyncio.run(client())
    finally:
        server.stop()
    assert not server._thread.is_alive()


def test_bind_error_is_reported():
    first = TelemetryServer('127.0.0.1', 0)
    first.start()
    second = TelemetryServer('127.0.0.1', first.port)
    try:
        with pytest.raises(RuntimeError, match='failed'):
            second.start()
    finally:
        first.stop()
        second.stop()


def test_fixed_pipeline_cannot_be_mislabelled_by_mode_control():
    state=TelemetryState(mode='slam',mode_locked=True)
    server=TelemetryServer('127.0.0.1',0,state=state)
    server._handle_message(json.dumps({'type':'control','action':'set_mode','mode':'vo'}))
    assert state.mode=='slam'


def test_final_frame_metadata_is_delivered_before_shutdown():
    state = TelemetryState(overlay_enabled=False)
    server = TelemetryServer('127.0.0.1', 0, state=state)
    server.start()

    async def client():
        async with websockets.connect(f'ws://127.0.0.1:{server.port}') as socket:
            final = make_frame_message(frame_index=9, timestamp=1., pose_T_wc=np.eye(4),
                                       tracking={}, map_state={}, sequence='03', total_frames=10,
                                       run_id='test-run', state=state)
            server.publish(final)
            await asyncio.to_thread(server.stop)
            frame = json.loads(await asyncio.wait_for(socket.recv(), 2))
            assert frame['frame_index'] == 9
            assert frame['sequence'] == '03'
            assert frame['total_frames'] == 10
            assert frame['run_id'] == 'test-run'
            assert not frame['overlay_enabled']
    try:
        asyncio.run(client())
    finally:
        server.stop()
    assert not server._thread.is_alive()
