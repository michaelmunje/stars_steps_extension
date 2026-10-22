import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import torch
import pyarrow as pa
import pyarrow.parquet as pq

from baselines.eth_ucy import fold_datasets,fold_recording_partitions,protocol
from baselines.status import CURVE_REFRESH_SECONDS,RunStatus,checkpoint_metadata
from baselines.provenance import snapshot_trajectron_shards
from baselines.trajectron_cache import Shard,load_index,recording_names
from baselines.device import training_device
from config import Config,SeedConfig,seed_everything
from data import scene_collate

ROOT=Path(__file__).resolve().parents[1]


class BaselineIntegrationTests(unittest.TestCase):
    def require_v3_data(self):
        path=next((ROOT/"datasets/eth/converted").glob("*.traj.parquet"),None)
        if path is None:self.skipTest("ETH data unavailable")
        metadata={k.decode():v.decode() for k,v in (pq.ParquetFile(path).schema_arrow.metadata or {}).items()}
        if metadata.get("canonical_schema_version")!="3":self.skipTest("baseline fixture is legacy v2")
    def test_cpu_requires_explicit_request(self):
        self.assertEqual(str(training_device("cpu")),"cpu")

    def test_five_location_protocol(self):
        self.require_v3_data()
        definition=protocol()
        expected={
            "eth": {"seq_eth"},
            "hotel": {"seq_hotel"},
            "univ": {"students001", "students003"},
            "zara1": {"crowds_zara01"},
            "zara2": {"crowds_zara02"},
        }
        self.assertEqual({key:set(value) for key,value in definition["folds"].items()},expected)
        self.assertEqual(definition["auxiliary_train_recordings"],[])
        self.assertEqual(
            set(definition["excluded_recordings"]),
            {"arxiepiskopi1", "crowds_zara03", "uni_examples"},
        )
        self.assertEqual(definition["folds"]["univ"],["students001","students003"])
        self.assertEqual(definition["folds"]["zara2"],["crowds_zara02"])
        self.assertEqual(definition["default_validation_mode"],"none")
        for fold in definition["folds"]:
            train,val,test=fold_datasets(fold,Config(train_rotation=False),"grouped")
            self.assertTrue(len(train));self.assertTrue(len(val));self.assertTrue(len(test))
            recordings=lambda part:{row["recording"] for dataset in part.datasets for row in dataset.rows}
            train_recordings,val_recordings,test_recordings=map(recordings,(train,val,test))
            self.assertEqual(test_recordings,expected[fold])
            self.assertFalse(train_recordings&val_recordings)
            self.assertFalse(train_recordings&test_recordings)
            self.assertFalse(val_recordings&test_recordings)
            full_train,no_val,full_test=fold_datasets(fold,Config(train_rotation=False),"none")
            self.assertIsNone(no_val)
            expected_train=set().union(*(set(value) for key,value in expected.items() if key!=fold))
            self.assertEqual(recordings(full_train),expected_train)
            self.assertEqual(recordings(full_test),expected[fold])

    def test_grouped_validation_is_location_disjoint_from_training(self):
        """N-029: grouped validation must reserve a whole scene_id location,
        not just one recording, so e.g. students001/students003 (both "univ")
        cannot land on opposite sides of train/val.

        Locations are hand-derived from ucy/convert.py's SCENE_IDS table and
        eth/source.json's scene_identity, independent of the implementation
        under test.
        """
        self.require_v3_data()
        location_by_recording = {
            "seq_eth": "eth",
            "seq_hotel": "hotel",
            "crowds_zara01": "zara1",
            "crowds_zara02": "zara2",
            "students001": "univ",
            "students003": "univ",
        }
        for fold in ("eth", "hotel", "univ", "zara1", "zara2"):
            partitions = fold_recording_partitions(fold, data_seed=42, validation_mode="grouped")
            train_locations = {location_by_recording[recording] for recording in partitions["train"]}
            val_locations = {location_by_recording[recording] for recording in partitions["val"]}
            self.assertFalse(
                train_locations & val_locations,
                f"fold {fold}: train/val share location(s) {train_locations & val_locations}",
            )
            # univ is a two-recording location: if it is selected for
            # validation, both students001 and students003 must move together.
            if any(location_by_recording[recording] == "univ" for recording in partitions["val"]):
                self.assertEqual(partitions["val"] & {"students001", "students003"}, {"students001", "students003"})

    def test_status_replaces_nonfinite_numbers_with_null(self):
        with tempfile.TemporaryDirectory() as tmp:
            status=RunStatus(Path(tmp),{"model":"test"})
            status.update(best=float("inf"))
            self.assertIsNone(json.loads((Path(tmp)/"status.json").read_text())["best"])
            status.metric({"stage":"train","iteration":1,"loss":2.0})
            status.metric({"stage":"train","iteration":2,"loss":1.0})
            status.update(status="complete")
            self.assertTrue((Path(tmp)/"training.log").exists())
            self.assertTrue((Path(tmp)/"curves/training_curves.png").exists())
            for line in (Path(tmp)/"training.log").read_text().splitlines():json.loads(line)

    def test_status_running_update_also_refreshes_curves(self):
        """MT-004 regression: an interrupted run (killed before 'complete'/'failed') must still
        get periodic curve refreshes from the 'running' status updates the trainers already emit
        once per epoch, per experimentation.md's "refresh curves periodically" requirement."""
        with tempfile.TemporaryDirectory() as tmp:
            status=RunStatus(Path(tmp),{"model":"test"})
            status.metric({"stage":"train","iteration":1,"loss":2.0})
            curves=Path(tmp)/"curves/training_curves.png"
            self.assertFalse(curves.exists())
            status.update(status="running",epoch=1)
            self.assertTrue(curves.exists(),"plot_curves must be invoked on status='running' updates")

    def test_status_running_curve_refresh_is_wall_clock_throttled(self):
        """E-013 regression: two trainers call update(status='running') from inside their batch
        loop, so an unthrottled render (~0.9 s of matplotlib) runs far more often than needed.
        A second immediate 'running' update must reuse the figures already on disk."""
        with tempfile.TemporaryDirectory() as tmp:
            status=RunStatus(Path(tmp),{"model":"test"})
            status.metric({"stage":"train","iteration":1,"loss":2.0})
            curves=Path(tmp)/"curves/training_curves.png"
            status.update(status="running",epoch=1,iteration=1)
            self.assertTrue(curves.exists())
            first=curves.stat().st_mtime_ns
            status.metric({"stage":"train","iteration":2,"loss":1.5})
            status.update(status="running",epoch=1,iteration=2)
            self.assertEqual(curves.stat().st_mtime_ns,first,
                             "an immediately following 'running' update must not re-render curves")
            # The throttle is wall-clock based, so the render resumes once the interval elapses;
            # this bounds curve staleness for a hard-killed run at CURVE_REFRESH_SECONDS.
            status._last_curve_time-=CURVE_REFRESH_SECONDS
            status.update(status="running",epoch=1,iteration=3)
            self.assertGreater(curves.stat().st_mtime_ns,first,
                               "curves must refresh again once CURVE_REFRESH_SECONDS has elapsed")

    def test_terminal_status_updates_are_never_throttled(self):
        """'complete'/'failed' must render unconditionally, even immediately after a render, so a
        finished or crashed run always leaves final curves matching its final metrics."""
        with tempfile.TemporaryDirectory() as tmp:
            status=RunStatus(Path(tmp),{"model":"test"})
            status.metric({"stage":"train","iteration":1,"loss":2.0})
            curves=Path(tmp)/"curves/training_curves.png"
            status.update(status="running",epoch=1,iteration=1)
            previous=curves.stat().st_mtime_ns
            for terminal in ("failed","complete"):
                status.metric({"stage":"train","iteration":2,"loss":1.0})
                status.update(status=terminal,epoch=1,iteration=2)
                current=curves.stat().st_mtime_ns
                self.assertGreater(current,previous,f"status={terminal!r} must render unthrottled")
                previous=current

    def test_running_update_without_metrics_does_not_consume_throttle_window(self):
        """A trainer emits 'running' updates during setup, before metrics.jsonl exists, where
        plot_curves has nothing to render. Those must not start the throttle clock and delay the
        first real curve refresh."""
        with tempfile.TemporaryDirectory() as tmp:
            status=RunStatus(Path(tmp),{"model":"test"})
            status.update(status="running",stage="preparing_cache_shards",iteration=0)
            self.assertIsNone(status._last_curve_time)
            status.metric({"stage":"train","iteration":1,"loss":2.0})
            status.update(status="running",stage="training",iteration=1)
            self.assertTrue((Path(tmp)/"curves/training_curves.png").exists(),
                            "the first 'running' update with plottable metrics must render")

    def test_checkpoint_metadata_reads_config_once_per_run(self):
        """MT-003 regression: config.json is immutable after RunStatus.__init__ writes it, so
        repeated checkpoint_metadata(run_dir) calls within a run must not re-read/re-hash it
        from disk on every call."""
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmp:
            run_dir=Path(tmp)
            RunStatus(run_dir,{"model":"test"})
            first=checkpoint_metadata(run_dir)
            original_read_bytes=Path.read_bytes
            calls=[]
            def counting_read_bytes(self):
                calls.append(self)
                return original_read_bytes(self)
            with patch.object(Path,"read_bytes",counting_read_bytes):
                second=checkpoint_metadata(run_dir)
                third=checkpoint_metadata(run_dir)
            self.assertEqual(first,second);self.assertEqual(first,third)
            self.assertEqual(calls,[],"checkpoint_metadata must not re-read config.json once cached")
            config=json.loads((run_dir/"config.json").read_text())
            self.assertEqual(first["run_config"],config)
            self.assertEqual(
                first["run_config_sha256"],
                "sha256:"+__import__("hashlib").sha256((run_dir/"config.json").read_bytes()).hexdigest(),
            )

    def test_status_legacy_normalization_matches_resolved_coordinate_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir=Path(tmp)
            RunStatus(run_dir,{"model":"test","coordinate_frame":"scene_mean_center"})
            config=json.loads((run_dir/"config.json").read_text())
            self.assertEqual(config["coordinate_frame"],"scene_mean_center")
            self.assertEqual(config["model_normalization"],"scene_mean_center")

    def test_status_records_git_commit_and_dirty_flag(self):
        expected_commit=subprocess.run(
            ["git","rev-parse","HEAD"],cwd=str(ROOT),capture_output=True,text=True,check=True,
        ).stdout.strip()
        with tempfile.TemporaryDirectory() as tmp:
            RunStatus(Path(tmp),{"model":"test"})
            config=json.loads((Path(tmp)/"config.json").read_text())
        self.assertIn("git_commit",config)
        self.assertIn("git_dirty",config)
        self.assertEqual(config["git_commit"],expected_commit)
        self.assertIsInstance(config["git_dirty"],bool)

    def test_status_records_lightweight_environment_info(self):
        with tempfile.TemporaryDirectory() as tmp:
            RunStatus(Path(tmp),{"model":"test"})
            config=json.loads((Path(tmp)/"config.json").read_text())
        self.assertIn("python_version",config)
        self.assertIn("platform",config)
        self.assertIn("torch_version",config)
        self.assertIn("cuda_available",config)
        self.assertEqual(config["torch_version"],torch.__version__)

    def test_status_refuses_to_overwrite_existing_run_without_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir=Path(tmp)
            RunStatus(run_dir,{"model":"test"})
            with self.assertRaisesRegex(RuntimeError,"Refusing to overwrite"):
                RunStatus(run_dir,{"model":"test"})
            # Explicit resume is allowed to continue writing into the same run directory.
            RunStatus(run_dir,{"model":"test"},resume=True)

    def test_trajectron_shard_snapshot_records_complete_streaming_population(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cache=root/"cache";cache.mkdir();first=cache/"first.zst";second=cache/"second.zst"
            first.write_bytes(b"first");second.write_bytes(b"second")
            shards=[
                Shard(root,"recording:part0-2","train",first,17),
                Shard(root,"recording:part2-4","train",second,23),
            ]
            output=snapshot_trajectron_shards(root/"run","train",shards)
            rows=pq.read_table(output).to_pylist()
            self.assertEqual([row["recording_part"] for row in rows],["recording:part0-2","recording:part2-4"])
            self.assertEqual(sum(row["budget"] for row in rows),40)
            self.assertTrue(all(row["size_bytes"]>0 for row in rows))

    def test_trajectron_index_is_selected_by_recording_and_split(self):
        """Cache preparation must not materialize the full PTDPES index."""
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);indices=root/"indices";indices.mkdir()
            pq.write_table(
                pa.table({
                    "recording":["train-a","train-a","val-b","other"],
                    "split":["train","val","val","train"],
                    "start_step":[0,1,2,3],
                    "accepted":[True,True,True,True],
                    "track_kept":[True,True,True,True],
                    "eval_kept":[True,True,True,True],
                }),
                indices/"scenes-seed42.parquet",
            )
            rows=load_index(root,42,recording="train-a",role="train")
            self.assertEqual(rows[["recording","split","start_step"]].to_dict("records"),[
                {"recording":"train-a","split":"train","start_step":0},
            ])
            self.assertEqual(recording_names(root,"train",42),["other","train-a"])
            self.assertEqual(recording_names(root,"val",42),["train-a","val-b"])

    def test_trajectron_environment_uses_reference_layout(self):
        self.require_v3_data()
        from baselines.trajectron_data import fold_environment

        env=fold_environment("hotel","train",max_recordings=1)
        self.assertGreaterEqual(len(env.scenes),1)
        self.assertEqual(env.scenes[0].dt,0.4)
        self.assertTrue(env.scenes[0].nodes)
        self.assertEqual(env.scenes[0].nodes[0].data.shape[1],6)
        key=(env.NodeType.PEDESTRIAN,env.NodeType.PEDESTRIAN)
        self.assertEqual(env.attention_radius[key],3.0)

    def test_trajectron_evaluate_raises_on_empty_population(self):
        """An empty prediction population must fail loudly, not report null metrics as complete."""
        from baselines.evaluate_trajectron import evaluate

        class _StubNodeType:
            PEDESTRIAN = "PEDESTRIAN"

        class _StubEnv:
            NodeType = _StubNodeType
            scenes = [object()]  # non-empty, but max_scenes=0 slices it away before use

        with self.assertRaisesRegex(RuntimeError, "Trajectron\\+\\+ evaluation population is empty"):
            evaluate(None, _StubEnv(), max_scenes=0)

    def test_agentformer_evaluation_raises_on_empty_population(self):
        """An empty test population must fail loudly, not complete with null metrics.

        The test population is forced empty directly via ``sources()`` rather than via
        ``--max-scenes 0`` because this evaluator does not run ``AgentFormerSceneDataset``
        end-to-end here; see ``test_agentformer_scene_dataset_limit_zero_is_empty`` for
        ``limit=0`` coverage on the dataset itself.

        Runs in-process: agentformer's and trajectron's vendored ``model`` namespaces no longer
        collide (E-005), so this no longer needs the subprocess isolation the E-001 fix used as a
        workaround. See ``test_agentformer_import_does_not_collide_with_trajectron_namespace``
        for the dedicated namespace-collision regression coverage.
        """
        from unittest.mock import patch
        from baselines.agentformer import build
        from baselines import evaluate_agentformer

        with tempfile.TemporaryDirectory() as tmp:
            run_dir=Path(tmp)/"run";checkpoint=Path(tmp)/"checkpoint.pt"
            _,model=build(torch.device("cpu"),"eth")
            torch.save({"model":model.state_dict()},checkpoint)
            argv=["evaluate_agentformer.py","--fold","eth","--checkpoint",str(checkpoint),
                  "--model","vae","--run-dir",str(run_dir),"--device","cpu"]
            original_argv=sys.argv;sys.argv=argv
            try:
                with patch("baselines.evaluate_agentformer.sources",return_value=(None,None,[])):
                    with self.assertRaisesRegex(RuntimeError,"AgentFormer evaluation population is empty"):
                        evaluate_agentformer.main()
            finally:
                sys.argv=original_argv

    def test_agentformer_scene_dataset_limit_zero_is_empty(self):
        """Regression test for E-006: ``limit=0`` is an explicit empty limit, distinct from
        ``limit=None`` (no limit). A prior ``if limit else len(source)`` implementation treated
        ``0`` as falsy and silently fell back to the full, unlimited dataset.
        """
        from baselines.agentformer import AgentFormerSceneDataset

        source=list(range(5))
        self.assertEqual(len(AgentFormerSceneDataset(source,limit=0)),0)
        self.assertEqual(len(AgentFormerSceneDataset(source,limit=None)),len(source))
        self.assertEqual(len(AgentFormerSceneDataset(source,limit=3)),3)
        self.assertEqual(len(AgentFormerSceneDataset(source,limit=100)),len(source))

    def test_agentformer_import_does_not_collide_with_trajectron_namespace(self):
        """Regression test for E-005: agentformer and trajectron-plus-plus both vendor a
        top-level ``model`` package. ``agentformer.py`` scopes its synthetic ``model``/``utils``
        registration to the duration of its own import and restores whatever was previously
        registered, so neither reference package can permanently shadow the other regardless of
        which imports first in the same process.
        """
        # This test module already imports baselines.trajectron_cache at module scope (which
        # pulls in trajectron's real `model` package as a side effect), so `model` in sys.modules
        # is already trajectron's real package here -- exercising the "trajectron first" order.
        self.assertIn("trajectron-plus-plus",str(sys.modules["model"].__path__[0]))
        import baselines.agentformer as agentformer
        self.assertTrue(hasattr(agentformer,"AgentFormer"))
        self.assertTrue(hasattr(agentformer,"DLow"))
        # agentformer's import must not leave its synthetic `model` package registered afterward.
        self.assertIn("trajectron-plus-plus",str(sys.modules["model"].__path__[0]))

        # The opposite order ("agentformer first") needs a clean process, since trajectron_cache
        # is already imported above before any test method runs.
        script=textwrap.dedent(f"""
            import sys
            sys.path.insert(0,{str(ROOT/"src")!r})
            import baselines.agentformer
            import baselines.trajectron_data
            print("OK")
        """)
        result=subprocess.run([sys.executable,"-c",script],capture_output=True,text=True,cwd=str(ROOT))
        self.assertEqual(result.returncode,0,f"stdout={result.stdout}\nstderr={result.stderr}")
        self.assertIn("OK",result.stdout)

    def _parsed_max_val_batches(self,module_name,argv,run_status_target):
        """Runs a trainer's ``main`` just far enough to capture the parsed ``--max-val-batches``
        value, by making the first ``RunStatus(...)`` call raise with the config dict it was
        given (E-011 regression: this flag used to not exist, capping validation at a hardcoded
        10 batches with no way to discover or change it).
        """
        from unittest.mock import patch
        import importlib
        module=importlib.import_module(module_name)
        captured={}
        def fake_run_status(run_dir,config):
            captured.update(config);raise RuntimeError("stop after parsing")
        original_argv=sys.argv;sys.argv=argv
        try:
            with patch(run_status_target,side_effect=fake_run_status):
                with self.assertRaisesRegex(RuntimeError,"stop after parsing"):
                    module.main()
        finally:
            sys.argv=original_argv
        self.assertIn("max_val_batches",captured)
        return captured["max_val_batches"]

    def test_agentformer_max_val_batches_defaults_to_ten(self):
        value=self._parsed_max_val_batches(
            "baselines.train_agentformer",
            ["train_agentformer.py","--stage","pretrain","--run-dir","/tmp/unused-run-dir","--device","cpu"],
            "baselines.train_agentformer.RunStatus",
        )
        self.assertEqual(value,10)

    def test_agentformer_max_val_batches_is_configurable(self):
        value=self._parsed_max_val_batches(
            "baselines.train_agentformer",
            ["train_agentformer.py","--stage","pretrain","--run-dir","/tmp/unused-run-dir","--device","cpu","--max-val-batches","3"],
            "baselines.train_agentformer.RunStatus",
        )
        self.assertEqual(value,3)

    def test_agentformer_dlow_max_val_batches_defaults_to_ten(self):
        with tempfile.TemporaryDirectory() as tmp:
            predictor_checkpoint=Path(tmp)/"predictor.pt";predictor_checkpoint.write_bytes(b"")
            value=self._parsed_max_val_batches(
                "baselines.train_agentformer_dlow",
                ["train_agentformer_dlow.py","--stage","pretrain","--run-dir",str(Path(tmp)/"run"),
                 "--predictor-checkpoint",str(predictor_checkpoint),"--device","cpu"],
                "baselines.train_agentformer_dlow.RunStatus",
            )
        self.assertEqual(value,10)

    def test_agentformer_dlow_max_val_batches_is_configurable(self):
        with tempfile.TemporaryDirectory() as tmp:
            predictor_checkpoint=Path(tmp)/"predictor.pt";predictor_checkpoint.write_bytes(b"")
            value=self._parsed_max_val_batches(
                "baselines.train_agentformer_dlow",
                ["train_agentformer_dlow.py","--stage","pretrain","--run-dir",str(Path(tmp)/"run"),
                 "--predictor-checkpoint",str(predictor_checkpoint),"--device","cpu","--max-val-batches","3"],
                "baselines.train_agentformer_dlow.RunStatus",
            )
        self.assertEqual(value,3)
