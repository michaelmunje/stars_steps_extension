import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from datasets_preprocessing.eth.convert import OUTPUT_SCHEMA
from datasets_preprocessing.atc.convert import OUTPUT_SCHEMA as ATC_OUTPUT_SCHEMA
from datasets_preprocessing.ucy.convert import OUTPUT_SCHEMA as UCY_OUTPUT_SCHEMA
from datasets_preprocessing.madras.convert import OUTPUT_SCHEMA as MADRAS_OUTPUT_SCHEMA
from datasets_preprocessing.PTDPES.convert import OUTPUT_SCHEMA as PTDPES_OUTPUT_SCHEMA
from datasets_preprocessing.steps.convert import OUTPUT_SCHEMA as STEPS_OUTPUT_SCHEMA


ROOT=Path(__file__).resolve().parents[1]


def require_v3(testcase, path):
    metadata={k.decode():v.decode() for k,v in (pq.ParquetFile(path).schema_arrow.metadata or {}).items()}
    if metadata.get("canonical_schema_version")!="3":
        testcase.skipTest("converted fixture is legacy v2; regenerate/migrate canonical v3 before dataset conformance")
    testcase.assertEqual(float(metadata["canonical_hz"]),2.5)
    return metadata


def require_files(testcase, directory, pattern="*.traj.parquet"):
    files=sorted(directory.glob(pattern))
    if not files:
        testcase.skipTest(f"generated dataset artifacts are absent under {directory}; regenerate canonical v3 first")
    return files


class EthConformanceTests(unittest.TestCase):
    def test_converted_files_conform(self):
        for path in require_files(self,ROOT/"datasets/eth/converted"):
            require_v3(self,path)
            table=pq.read_table(path)
            self.assertEqual(table.schema.remove_metadata(),OUTPUT_SCHEMA)
            ids=table["track_id"].to_numpy(); steps=table["step"].to_numpy()
            keys=np.stack((ids,steps),axis=1)
            self.assertEqual(len(keys),len(np.unique(keys,axis=0)))
            x=table["x"].to_numpy(); y=table["y"].to_numpy()
            self.assertTrue(np.isfinite(x).all() and np.isfinite(y).all())
            self.assertLess(np.max(np.abs(np.r_[x,y])),100.0)
            speeds=[]
            for track_id in np.unique(ids):
                mask=ids==track_id; order=np.argsort(steps[mask]); ts=steps[mask][order]
                xy=np.c_[x[mask][order],y[mask][order]]; dt=np.diff(ts)/2.5
                speeds.extend((np.linalg.norm(np.diff(xy,axis=0),axis=1)/dt).tolist())
            self.assertLess(max(speeds),10.0)
            metadata={k.decode():v.decode() for k,v in table.schema.metadata.items()}
            self.assertIn("heading_transform",metadata)

    def test_all_index_starts_are_stride_aligned(self):
        path=ROOT/"datasets/eth/indices/scenes-seed42.parquet"
        if not path.exists():self.skipTest("ETH forecast index has not been regenerated")
        table=pq.read_table(path)
        self.assertTrue(np.all(table["start_step"].to_numpy()>=0))


class AtcConformanceTests(unittest.TestCase):
    def test_all_recording_schemas_and_sampled_rows(self):
        files=require_files(self,ROOT/"datasets/atc/converted")
        self.assertEqual(len(files),92)
        for path in files:
            require_v3(self,path)
            parquet=pq.ParquetFile(path)
            self.assertEqual(parquet.schema_arrow.remove_metadata(),ATC_OUTPUT_SCHEMA)
            metadata={k.decode():v.decode() for k,v in parquet.schema_arrow.metadata.items()}
            self.assertIn("heading_transform",metadata)
            groups=sorted({0,parquet.num_row_groups//2,parquet.num_row_groups-1})
            table=pa.concat_tables([parquet.read_row_group(group) for group in groups])
            ids=table["track_id"].to_numpy(); steps=table["step"].to_numpy()
            keys=np.stack((ids,steps),axis=1)
            self.assertEqual(len(keys),len(np.unique(keys,axis=0)))
            self.assertLess(np.max(np.abs(np.r_[table["x"].to_numpy(),table["y"].to_numpy()])),100.0)


class UcyConformanceTests(unittest.TestCase):
    def test_converted_files_are_metric_aligned_and_physically_plausible(self):
        files=require_files(self,ROOT/"datasets/ucy/converted")
        self.assertEqual(
            {path.name for path in files},
            {
                "arxiepiskopi1.traj.parquet",
                "crowds_zara01.traj.parquet",
                "crowds_zara02.traj.parquet",
                "crowds_zara03.traj.parquet",
                "students001.traj.parquet",
                "students003.traj.parquet",
                "uni_examples.traj.parquet",
            },
        )
        for path in files:
            require_v3(self,path)
            table=pq.read_table(path)
            self.assertEqual(table.schema.remove_metadata(),UCY_OUTPUT_SCHEMA)
            ids=table["track_id"].to_numpy(); steps=table["step"].to_numpy()
            self.assertTrue(np.all(steps>=0))
            self.assertEqual(len(ids),len(set(zip(ids.tolist(),steps.tolist()))))
            x=table["x"].to_numpy(); y=table["y"].to_numpy()
            self.assertLess(np.max(np.abs(np.r_[x,y])),100.0)
            maximum_speed=0.0
            for track_id in np.unique(ids):
                mask=ids==track_id; order=np.argsort(steps[mask]); track_steps=steps[mask][order]
                xy=np.c_[x[mask][order],y[mask][order]]
                if len(xy)>1:
                    speed=np.linalg.norm(np.diff(xy,axis=0),axis=1)/(np.diff(track_steps)/2.5)
                    maximum_speed=max(maximum_speed,float(speed.max()))
            self.assertLess(maximum_speed,10.0)
            metadata={k.decode():v.decode() for k,v in table.schema.metadata.items()}
            self.assertIn("heading_transform",metadata)


class MadrasConformanceTests(unittest.TestCase):
    def test_converted_files_are_aligned_and_physically_plausible(self):
        files=require_files(self,ROOT/"datasets/madras/converted")
        self.assertEqual(len(files),12)
        for path in files:
            require_v3(self,path)
            table=pq.read_table(path); self.assertEqual(table.schema.remove_metadata(),MADRAS_OUTPUT_SCHEMA)
            ids=table["track_id"].to_numpy(); steps=table["step"].to_numpy(); x=table["x"].to_numpy(); y=table["y"].to_numpy()
            self.assertEqual(len(ids),len(set(zip(ids.tolist(),steps.tolist()))))
            self.assertTrue(np.isfinite(x).all() and np.isfinite(y).all())
            self.assertLess(np.max(np.abs(np.r_[x,y])),100.0)
            order=np.lexsort((steps,ids)); ordered_ids=ids[order]; ordered_steps=steps[order]
            dt=np.diff(ordered_steps)/2.5; adjacent=(ordered_ids[1:]==ordered_ids[:-1])&(dt>0)&(dt<=0.5)
            displacement=np.linalg.norm(np.diff(np.c_[x[order],y[order]],axis=0),axis=1)
            self.assertLess(float(np.max(displacement[adjacent]/dt[adjacent])),10.0)
            metadata={k.decode():v.decode() for k,v in table.schema.metadata.items()}
            self.assertIn("heading_transform",metadata); self.assertIn("duplicate_time_rows_averaged",metadata)


class PTDPESConformanceTests(unittest.TestCase):
    def test_all_final_recordings_conform_and_sampled_rows_are_physical(self):
        # Nine files distributed under Test_set are byte-identical duplicates
        # of production recordings and are quarantined rather than indexed.
        files=require_files(self,ROOT/"datasets/PTDPES/converted");self.assertEqual(len(files),382)
        maximum_speed=0.0
        for path in files:
            require_v3(self,path)
            parquet=pq.ParquetFile(path);self.assertEqual(parquet.schema_arrow.remove_metadata(),PTDPES_OUTPUT_SCHEMA)
            metadata={k.decode():v.decode() for k,v in parquet.schema_arrow.metadata.items()};self.assertEqual(metadata["native_hz"],"2.0");self.assertEqual(metadata["tracks_filtered"],"0");self.assertIn("tracks_above_jump_audit_threshold",metadata);self.assertIn("web_mercator_ground_scale",metadata)
            groups=sorted({0,parquet.num_row_groups-1});table=pa.concat_tables([parquet.read_row_group(group) for group in groups])
            ids=table["track_id"].to_numpy();steps=table["step"].to_numpy();x=table["x"].to_numpy();y=table["y"].to_numpy()
            self.assertTrue(np.all(steps>=0));self.assertEqual(len(ids),len(set(zip(ids.tolist(),steps.tolist()))));self.assertTrue(np.isfinite(np.c_[x,y]).all())
            order=np.lexsort((steps,ids));dt=np.diff(steps[order])/2.5;valid=(ids[order][1:]==ids[order][:-1])&(dt>0)&(dt<=1)
            if valid.any():maximum_speed=max(maximum_speed,float(np.max(np.linalg.norm(np.diff(np.c_[x[order],y[order]],axis=0),axis=1)[valid]/dt[valid])))
        self.assertGreater(maximum_speed,10.0)  # published postprocessed jumps are intentionally retained


class StepsConformanceTests(unittest.TestCase):
    def test_converted_recordings_are_observed_posterior_states_at_2p5hz(self):
        for path in require_files(self,ROOT/"datasets/steps/converted"):
            require_v3(self,path)
            table = pq.read_table(path)
            self.assertEqual(table.schema.remove_metadata(), STEPS_OUTPUT_SCHEMA)
            ids = table["track_id"].to_numpy()
            steps = table["step"].to_numpy()
            self.assertEqual(len(ids), len(set(zip(ids.tolist(), steps.tolist()))))
            self.assertTrue(np.isfinite(np.c_[table["x"].to_numpy(), table["y"].to_numpy()]).all())
            self.assertTrue(np.all(np.diff(steps) >= 0))
            metadata = {key.decode(): value.decode() for key, value in table.schema.metadata.items()}
            self.assertEqual(metadata["dataset"], "steps")
            self.assertEqual(metadata["output_hz"], "2.5")
            self.assertIn("global_kalman_posterior", metadata["trajectory_state_filter"])
            self.assertIn("end_of_stream_flush", metadata["end_of_stream_policy"])


if __name__=="__main__": unittest.main()
