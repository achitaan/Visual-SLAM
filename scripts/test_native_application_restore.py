"""Exercise restoration against real frozen bytes and hostile/mismatched paths."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from restore_native_application_bindings import PACKAGE, ROOT, restore


class RestoreTests(unittest.TestCase):
    def setUp(self):
        cache = ROOT/'.cache'
        cache.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='native-restore-test-',dir=cache)
        self.root = Path(self.temp.name).resolve()
        self.assertTrue(self.root.is_relative_to(cache.resolve()))
        self.addCleanup(self.temp.cleanup)

    def fixture(self):
        package = self.root/'package'
        (package/'campaign').mkdir(parents=True)
        data = (PACKAGE/'campaign/native_application_control_ordered_v6.py').read_bytes()
        (package/'campaign/control.py').write_bytes(data)
        (package/'local-bindings.json').write_text(json.dumps({'files':{'control.py':{
            'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}}}))
        return package, data

    def test_real_frozen_closure_restores_exactly_and_is_idempotent(self):
        target = self.root/'restored'
        result = restore(PACKAGE,target,apply=True)
        expected_count = len(json.loads((PACKAGE/'local-bindings.json').read_text())['files'])
        self.assertGreaterEqual(expected_count,56)
        self.assertEqual(result['restored_files'],expected_count)
        result = restore(PACKAGE,target,apply=True)
        self.assertEqual(result['missing_files'],0)
        records = json.loads((PACKAGE/'local-bindings.json').read_text())['files']
        for name in records:
            self.assertEqual((target/name).read_bytes(),(PACKAGE/'campaign'/name).read_bytes())

    def test_check_only_creates_nothing(self):
        package,_ = self.fixture()
        target = self.root/'absent'
        self.assertEqual(restore(package,target)['missing_files'],1)
        self.assertFalse(target.exists())

    def test_existing_difference_is_preserved(self):
        package,_ = self.fixture()
        target = self.root/'existing';target.mkdir()
        (target/'control.py').write_bytes(b'user work')
        with self.assertRaisesRegex(ValueError,'Preserving existing'):
            restore(package,target,apply=True)
        self.assertEqual((target/'control.py').read_bytes(),b'user work')

    def test_changed_package_refused_before_destination_creation(self):
        package,_ = self.fixture()
        (package/'campaign/control.py').write_bytes(b'corrupt')
        target = self.root/'absent'
        with self.assertRaisesRegex(ValueError,'hash mismatch'):
            restore(package,target,apply=True)
        self.assertFalse(target.exists())

    def test_manifest_traversal_refused(self):
        package,data = self.fixture()
        (package/'escape.py').write_bytes(data)
        (package/'local-bindings.json').write_text(json.dumps({'files':{'../escape.py':{
            'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}}}))
        with self.assertRaisesRegex(ValueError,'escaped root'):
            restore(package,self.root/'destination',apply=True)
        self.assertFalse((self.root/'destination').exists())


if __name__ == '__main__':
    unittest.main()
