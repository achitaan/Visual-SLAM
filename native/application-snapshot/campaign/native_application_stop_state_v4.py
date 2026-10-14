"""Atomic completion is admitted only after an exact terminal whole-exit seal."""
from native_application_stop_state_v3 import ApplicationStopState as FrozenState


class ApplicationStopState(FrozenState):
    def __init__(self, run_id, expected_frames):
        super().__init__(run_id, expected_frames)
        self.completed_audit_verified = False

    def mark_complete(self, audit):
        if (not isinstance(audit, dict)
                or audit.get('status') != 'sealed_all_admitted_groups_and_token_orphans_exited'
                or audit.get('run_id') != self.run_id
                or audit.get('gate_purpose') != 'completed'
                or audit.get('all_owned_exited') is not True
                or audit.get('seal', {}).get('all_owned_exited') is not True):
            raise ValueError('normal completion requires exact completed whole-exit seal')
        with self._lock:
            if self.phase in ('stopping', 'cancelled'):
                return False
            if self.phase != 'running':
                raise RuntimeError('completion requires running application')
            self.completed_audit_verified = True
            self.phase = 'completed'
            self.last_status = 'completed_whole_owned_cleanup_confirmed'
            self._worker_control = None
            return True

    def snapshot_locked(self):
        value = super().snapshot_locked()
        if self.phase == 'completed':
            value['cleanup_confirmed'] = self.completed_audit_verified
        return value
