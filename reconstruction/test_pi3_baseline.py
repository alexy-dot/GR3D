#!/usr/bin/env python3

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from reconstruction.run_pi3_baseline import resolve_pi3_root


class Pi3BaselineTests(unittest.TestCase):
    def test_default_pi3_root_is_inside_project(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            self.assertEqual(
                resolve_pi3_root(project, {}),
                (project / "third_party" / "Pi3").resolve(),
            )

    def test_environment_selects_external_clone(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            external = Path(directory) / "Pi3"
            self.assertEqual(
                resolve_pi3_root(Path("unused"), {"PI3_ROOT": str(external)}),
                external.resolve(),
            )


if __name__ == "__main__":
    unittest.main()
