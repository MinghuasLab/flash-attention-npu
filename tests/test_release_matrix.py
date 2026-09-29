import importlib.util
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "ci" / "release_matrix.py"
SPEC = importlib.util.spec_from_file_location("_test_release_matrix", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ReleaseMatrixTest(unittest.TestCase):
    def test_repository_matrix_has_unique_910_and_950_rows(self):
        rows = MODULE.load_matrix(REPO_ROOT / "ci" / "build_matrix.tsv")
        self.assertEqual({row["npu"] for row in rows}, {"910", "950"})
        self.assertEqual(len({row["name"] for row in rows}), len(rows))

    def test_invalid_field_count_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            matrix = Path(directory) / "matrix.tsv"
            matrix.write_text("invalid|row\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "expected 12 fields"):
                MODULE.load_matrix(matrix)

    def test_v2_950_combination_is_rejected(self):
        row = "name|base|cp312|2.9.0|2.9.0.post2|release|9.1|950|v2|detected|x86_64|\n"
        with tempfile.TemporaryDirectory() as directory:
            matrix = Path(directory) / "matrix.tsv"
            matrix.write_text(row, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "v2 has no 950 backend"):
                MODULE.load_matrix(matrix)


if __name__ == "__main__":
    unittest.main()
