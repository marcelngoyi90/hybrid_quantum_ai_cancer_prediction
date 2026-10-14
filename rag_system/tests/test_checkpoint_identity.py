from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.joint_checkpointing import calculate_json_sha256


class CheckpointIdentityTests(unittest.TestCase):
    def test_json_hash_is_independent_of_platform_line_endings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unix_file = root / "unix.json"
            windows_file = root / "windows.json"
            content = '{\n  "version": "2.0"\n}\n'

            unix_file.write_bytes(content.encode("utf-8"))
            windows_file.write_bytes(
                content.replace("\n", "\r\n").encode("utf-8")
            )

            self.assertEqual(
                calculate_json_sha256(unix_file),
                calculate_json_sha256(windows_file),
            )


if __name__ == "__main__":
    unittest.main()
