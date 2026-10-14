"""Application prefix-delivery successor; no estimator/process ownership changes.

The frozen follower still decodes, emits and verifies immutable original-ID
coverage and lifecycle. Deferred reads are re-polled, never returned as stale
successful observations. The native driver must enforce its owned absolute cap.
"""
from native_app_follower import NativeAppFollower as FrozenAppFollower
from native_observation_delivery import DeliveryBudget, ObservationDeferred


class NativeAppFollower(FrozenAppFollower):
    def __init__(self, *args, read_snapshot=None, **kwargs):
        super().__init__(*args, read_snapshot=read_snapshot, **kwargs)
        self._budget = DeliveryBudget(clock=self.clock, sleep=self.sleep)
        for reader in self._readers.values():
            reader.budget = self._budget

    def _prefix(self, name, deadline, poll):
        while True:
            try:
                return super()._prefix(name, deadline, poll)
            except ObservationDeferred:
                if self.clock() >= deadline:
                    raise TimeoutError('absolute application observation deadline exceeded')
                self.sleep(self.poll_s)

    @property
    def observation(self):
        result = super().observation
        result.update(delayed_returns=self._budget.delayed_returns,
                      deferrals=self._budget.deferrals,
                      max_read_s=self._budget.max_read_s,
                      semantics='retry_admission_window_with_absolute_application_deadline')
        return result
