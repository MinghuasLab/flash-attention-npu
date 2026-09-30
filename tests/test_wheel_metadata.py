import importlib.util
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from packaging.version import Version

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = REPO_ROOT / "flash_attn_npu" / "_wheel_metadata.py"
    spec = importlib.util.spec_from_file_location("_test_wheel_metadata", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


metadata = load_module()


class WheelMetadataTest(unittest.TestCase):
    ENVIRONMENT = {
        "FLASH_ATTN_BUILD_NPU": "910",
        "FLASH_ATTN_CANN_VERSION": "9.1",
        "FLASH_ATTN_TORCH_VERSION": "2.9.0",
        "FLASH_ATTN_TORCH_NPU_VERSION": "2.9.0.post2",
        "FLASH_ATTN_BUILD_VERSION": "all",
        "FLASH_ATTN_FORCE_CXX11_ABI": "FALSE",
        "FLASH_ATTN_PYTHON_TAG": "cp312",
        "FLASH_ATTN_PYTHON_ABI_TAG": "cp312",
        "FLASH_ATTN_PLATFORM_TAG": "linux_aarch64",
        "FLASH_ATTN_WHEEL_REPOSITORY": "example/project",
    }

    def config(self, **overrides):
        environment = dict(self.ENVIRONMENT)
        environment.update(overrides)
        with patch.dict(os.environ, environment, clear=True):
            return metadata.get_build_config(version="0.4.1")

    def test_wheel_filename_contains_compatibility_dimensions(self):
        config = self.config()
        self.assertEqual(
            metadata.get_wheel_filename(config),
            "flash_attn_npu-0.4.1+npu910cann91torch29torchnpu290post2apiallabifalse-"
            "cp312-cp312-linux_aarch64.whl",
        )

    def test_patch_versions_share_major_minor_compatibility_tokens(self):
        self.assertEqual(metadata.major_minor_token("9.1"), "91")
        self.assertEqual(metadata.major_minor_token("9.1.0"), "91")
        self.assertEqual(metadata.major_minor_token("2.9.0+cpu"), "29")

    def test_npu_targets_do_not_collide(self):
        wheel_910 = metadata.get_wheel_filename(self.config())
        wheel_950 = metadata.get_wheel_filename(
            self.config(FLASH_ATTN_BUILD_NPU="950", FLASH_ATTN_PLATFORM_TAG="linux_x86_64")
        )
        self.assertNotEqual(wheel_910, wheel_950)

    def test_release_url_uses_same_filename(self):
        config = self.config()
        self.assertEqual(
            metadata.get_release_url(config),
            f"https://github.com/example/project/releases/download/v0.4.1/"
            f"{metadata.get_wheel_filename(config)}",
        )

    def test_local_version_is_pep440_compatible(self):
        with patch.dict(
            os.environ,
            {**self.ENVIRONMENT, "FLASH_ATTN_LOCAL_VERSION": "dev-1"},
            clear=True,
        ):
            config = metadata.get_build_config(version="0.4.1")
        self.assertTrue(config.wheel_version.startswith("0.4.1+dev1.npu910"))
        self.assertEqual(str(Version(config.wheel_version)), config.wheel_version)


if __name__ == "__main__":
    unittest.main()
