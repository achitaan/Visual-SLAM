"""Guard and routing tests for the packaged entry, using its real frozen bindings."""
import contextlib
import io
import json
import unittest
from unittest.mock import patch

import run_native_candidate as entry


class PackagedEntryTests(unittest.TestCase):
    def setUp(self):
        capacity = patch.object(entry.original.shutil,'disk_usage',return_value=type('Disk',(),{'free':10**12})())
        capacity.start();self.addCleanup(capacity.stop)

    def test_all_four_prepared_presets_keep_exact_original_arguments(self):
        for preset in entry.LAUNCHERS:
            run = 'native_prototype_packaged_contract_'+preset
            original = entry.original.plan(preset,run)
            actual = entry.plan(preset,run)
            expected = list(original['command'])
            expected[2] = str(entry.original.BASE/entry.LAUNCHERS[preset])
            self.assertEqual(actual['command'],expected)
            self.assertEqual(actual['required_physical_free_bytes'],original['required_physical_free_bytes'])
            self.assertTrue(actual['control_ordering_repair'])

    def test_check_starts_no_process(self):
        with patch.object(entry.original,'assert_runtime_ready') as host, patch.object(entry.original,'run_foreground') as run, contextlib.redirect_stdout(io.StringIO()) as text:
            self.assertEqual(entry.main(['--preset','quick04','--check']),0)
        host.assert_not_called();run.assert_not_called()
        self.assertEqual(json.loads(text.getvalue())['packaged_binding_files_verified'],83)

    def test_changed_binding_manifest_refused(self):
        with patch.object(entry,'BINDINGS_SHA256','0'*64), self.assertRaisesRegex(ValueError,'manifest changed'):
            entry.plan('quick04','native_prototype_packaged_badmanifest')

    def test_changed_frontend_refused_before_setup_writes(self):
        with patch.object(entry,'FRONTEND_SHA256',{'scripts/online_candidate.py':'0'*64}), patch.object(entry,'restore') as restore, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(entry.main(['--preset','quick04','--setup','--check']),2)
        restore.assert_not_called()

    def test_original_scope_and_run_ownership_guards_still_apply(self):
        for preset, run in [('full05','native_prototype_packaged_invalid'),('quick04','../../outside')]:
            with self.assertRaises(ValueError):
                entry.plan(preset,run)

    def test_actual_saved_stop_result_is_accepted_by_terminal_bridge(self):
        value={'reviewed_inner_launcher':entry.LAUNCHERS['quick04'],
            'run_id':'native_prototype_repoapp_mapping_04_60_seed2_ordered30show','frames':60}
        result=entry.original.verify_terminal_result(value)
        # Original observed browser/native test, not a synthetic cleanup seal.
        self.assertEqual(result['emitted_original_ids'],list(range(40)))
        self.assertEqual(result['unprocessed_original_ids'],list(range(40,60)))


if __name__=='__main__':
    unittest.main()
