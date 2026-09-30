import importlib.util
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "ci" / "validate_release_version.py"
SPEC = importlib.util.spec_from_file_location("_test_release_version", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ReleaseVersionTest(unittest.TestCase):
    def test_current_version_matches_release_tag(self):
        version = MODULE.load_package_version(REPO_ROOT)
        MODULE.validate_release_version(f"v{version}", version)

    def test_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "does not match"):
            MODULE.validate_release_version("v9.9.9", "0.4.1")

    def test_non_version_tag_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must match"):
            MODULE.validate_release_version("release", "0.4.1")


if __name__ == "__main__":
    unittest.main()
