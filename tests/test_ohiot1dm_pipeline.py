"""Regression tests for the OhioT1DM preprocessing and model metadata path."""

from __future__ import annotations

import tempfile
import unittest
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from data.ohiot1dm_dataset import ProcessedOhioT1DMDataset
from models.st_msffnet import build_physiology_prior
from preprocess_ohiot1dm import (
    DEFAULT_NODE_NAMES,
    PreprocessConfig,
    discover_recordings,
    process_all_recordings,
    save_outputs,
)


class OhioT1DMPipelineTest(unittest.TestCase):
    def _write_recording(
        self,
        path: Path,
        subject_id: str,
        start: datetime,
        num_steps: int = 40,
    ) -> None:
        root = ET.Element("patient", id=subject_id)
        sections = {
            "glucose_level": lambda step: 100.0 + step,
            "basis_steps": lambda step: float(step % 4),
            "basis_gsr": lambda step: 0.001 + step * 0.00001,
            "basis_skin_temperature": lambda step: 86.0 + step * 0.01,
            "basis_heart_rate": lambda step: 65.0 + step % 8,
        }
        for section_name, value_fn in sections.items():
            section = ET.SubElement(root, section_name)
            for step in range(num_steps):
                timestamp = (start + timedelta(minutes=5 * step)).strftime(
                    "%d-%m-%Y %H:%M:%S"
                )
                ET.SubElement(
                    section,
                    "event",
                    ts=timestamp,
                    value=str(value_fn(step)),
                )
        ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)

    def test_preprocessing_preserves_recording_boundaries_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_root = root / "raw"
            output_dir = root / "processed"
            data_root.mkdir()
            self._write_recording(
                data_root / "559-ws-training.xml",
                "559",
                datetime(2021, 12, 7, 0, 0),
            )
            self._write_recording(
                data_root / "559-ws-testing.xml",
                "559",
                datetime(2022, 1, 7, 0, 0),
            )
            (data_root / ".~9-ws-testing.xml").write_text(
                "not XML", encoding="utf-8"
            )

            recordings = discover_recordings(data_root)
            self.assertEqual(len(recordings), 2)
            self.assertNotIn(".~9-ws-testing.xml", {item.path.name for item in recordings})

            cfg = PreprocessConfig(
                data_root=data_root,
                output_dir=output_dir,
                history_steps=4,
                horizon_steps=3,
            )
            x_cgm, x_physio, y, metadata, summary = process_all_recordings(cfg)
            self.assertEqual(x_cgm.shape, (68, 4, 1))
            self.assertEqual(x_physio.shape, (68, 4, 4))
            self.assertEqual(y.shape, (68, 3))
            self.assertTrue(np.isfinite(x_cgm).all())
            self.assertTrue(np.isfinite(x_physio).all())
            self.assertTrue(np.isfinite(y).all())
            self.assertEqual(tuple(DEFAULT_NODE_NAMES), (
                "activity",
                "gsr",
                "skin_temperature",
                "heart_rate",
            ))
            self.assertEqual(set(metadata["source_split"]), {"training", "testing"})
            self.assertTrue(
                metadata.groupby("source_file")["source_split"].nunique().eq(1).all()
            )
            self.assertTrue(summary["num_samples"].eq(34).all())

            save_outputs(cfg, x_cgm, x_physio, y, metadata, summary)
            dataset = ProcessedOhioT1DMDataset(output_dir, required_future_steps=3)
            self.assertEqual(len(dataset), 68)
            self.assertEqual(dataset.node_names, DEFAULT_NODE_NAMES)
            self.assertEqual(dataset.num_physio_nodes, 4)
            self.assertEqual(dataset[0]["future_glucose"].shape, (3,))

    def test_four_node_physiology_prior_uses_semantic_aliases(self) -> None:
        prior = build_physiology_prior(DEFAULT_NODE_NAMES)
        expected = np.asarray(
            [
                [0.00, 0.30, 0.00, 0.50],
                [0.30, 0.00, 0.40, 0.45],
                [0.00, 0.40, 0.00, 0.25],
                [0.50, 0.45, 0.25, 0.00],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(prior.numpy(), expected)


if __name__ == "__main__":
    unittest.main()
