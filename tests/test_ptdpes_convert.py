import tempfile,unittest
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from datasets_preprocessing.PTDPES import convert as ptdpes
ROOT=Path(__file__).resolve().parents[1]
class PTDPESTests(unittest.TestCase):
    def test_physical_scene_identity_is_not_collection_name(self):
        root=Path("/data/02.Trajectories_modified")
        comparative=root/"01.Trajectories_comparative"/"csv"/"09_saturday_Varberg_15fps_30min_yolov10l_final.csv"
        seasonal=root/"02.Trajectories_season"/"Varberg"/"speed"/"09_20250401_morning_Varberg_15_1800sec_HD_yolov10l_final_speed.csv"
        self.assertEqual(ptdpes.scene_id_from_path(comparative,root),"varberg")
        self.assertEqual(ptdpes.scene_id_from_path(seasonal,root),"varberg")
    def test_scene_name_variants_share_physical_identity(self):
        root=Path("/data/02.Trajectories_modified")
        directory=root/"01.Trajectories_comparative"/"csv"
        names=(
            "102_morning_Bielsko Biala_15fps_30min_yolov10l_final_speed.csv",
            "102_noon_Bielsko-Biala_15fps_30min_yolov10l_final_speed.csv",
            "102_saturday_Bieslko_Biala_15fps_30min_yolov10l_final_speed.csv",
        )
        self.assertEqual(
            {ptdpes.scene_id_from_path(directory/name,root) for name in names},
            {"bielsko_biala"},
        )
    def source_files_or_skip(self):
        try:
            return ptdpes.source_files(ROOT/"datasets/PTDPES")
        except FileNotFoundError as error:
            self.skipTest(str(error))

    def test_names_include_context(self):
        root=Path("r");self.assertNotEqual(ptdpes.recording_name(root/"a/x.csv",root),ptdpes.recording_name(root/"b/x.csv",root))
    def test_small_real_file(self):
        files=self.source_files_or_skip();source=min(files,key=lambda p:p.stat().st_size);root=ROOT/"datasets/PTDPES/raw/01.Trajectories/02.Trajectories_modified"
        with tempfile.TemporaryDirectory() as tmp:
            table=pq.read_table(ptdpes.convert_file(source,Path(tmp)/"x.parquet",root));self.assertEqual(table.schema.remove_metadata(),ptdpes.OUTPUT_SCHEMA)
            ids=table["track_id"].to_numpy();steps=table["step"].to_numpy()
            for track_id in np.unique(ids):
                track_steps=np.sort(steps[ids==track_id]);self.assertTrue(np.all(np.diff(track_steps)>0))
            self.assertTrue(any(np.any(np.diff(np.sort(steps[ids==track_id]))==1) for track_id in np.unique(ids)))
            self.assertEqual(table.schema.metadata[b"native_hz"],b"2.0");self.assertEqual(table.schema.metadata[b"output_hz"],b"2.5")
    def test_test_set_is_quarantined(self):
        files=self.source_files_or_skip()
        self.assertFalse(any("Test_set" in path.parts for path in files))
if __name__=="__main__":unittest.main()
