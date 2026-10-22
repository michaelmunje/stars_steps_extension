import unittest
import json
import math
import tempfile
from collections import Counter
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from area_calibration import calibrate_atc,calibration_status
from stats import AREA_CELL_SIZE_M,AREA_MASS_FRACTION,AREA_MIN_CELLS,STATS_SCHEMA_VERSION,_effective_grid_area,_effective_grid_area_from_counts,_geographic_polygon_area,_grid_cell_counts,_strided_step_occupancy,cv_fde,heading_entropy,min_time_to_closest_approach_scene,path_efficiency,rms_jerk


def _write_source_json(dataset_dir,area_m2=900.0,area_source="test fixture (Q-006)"):
    """calibrate_atc now reads the documented area from source.json (Q-006), so
    every synthetic dataset_dir it is called on needs one. area_m2=900.0 by
    default so the pre-existing ratio_to_documented/900.0 assertions below keep
    checking the same arithmetic they always did."""
    (dataset_dir/"source.json").write_text(json.dumps({"area_m2":area_m2,"area_source":area_source}))


def _linear_scene(positions,velocities,obs=8,dt=.4,steps=20):
    """Constant-velocity scene for min_time_to_closest_approach_scene, with agent 0 as the primary.

    ``positions[i]`` is agent ``i``'s location in metres at the last observed
    step ``obs-1`` and ``velocities[i]`` its velocity in m/s, so each test can
    state the forecast-origin geometry directly instead of back-computing it
    from a track. Returns ``(xy,valid)`` with ``xy`` shaped ``(steps,N,2)``.
    """
    offsets=(np.arange(steps)-(obs-1))[:,None]*dt
    xy=np.stack([np.asarray(p,float)+offsets*np.asarray(v,float) for p,v in zip(positions,velocities)],axis=1)
    return xy,np.ones(xy.shape[:2],bool)


def _mass_cells_reference(values,mass_fraction):
    """Smallest number of most-visited cells whose counts reach ``mass_fraction``
    of all observations, computed directly from the definition in Python."""
    target=mass_fraction*sum(values);cumulative=0
    for position,value in enumerate(sorted(values,reverse=True)):
        cumulative+=value
        if cumulative>=target: return position+1
    return len(values)


def _occupancy_reference(unique_steps,counts,stride):
    """Occupancy stated directly as the sampling grid it represents:
    one slot per step in ``range(min_step,max_step+1,stride)``."""
    by_step=dict(zip(unique_steps.tolist(),counts.tolist()))
    return np.asarray([by_step.get(step,0) for step in
                       range(int(unique_steps.min()),int(unique_steps.max())+1,stride)],float)


class StatisticsTests(unittest.TestCase):
    def test_schema_version_is_positive(self):
        self.assertGreaterEqual(STATS_SCHEMA_VERSION,1)

    def test_geographic_polygon_area(self):
        # A 0.001-degree square at the equator is about 111.2 m per side.
        square=np.array([[0,0],[.001,0],[.001,.001],[0,.001]],float)
        self.assertAlmostEqual(_geographic_polygon_area(square),12_364,delta=100)

    def test_effective_grid_area_rejects_strays(self):
        dense=np.repeat(np.array([[i*.5,0.] for i in range(40)]),20,axis=0)
        points=np.vstack((dense,[[10_000.,10_000.]]))
        # A single stray can move the discrete cumulative cutoff by at most one
        # cell, but cannot inflate the footprint by its geometric distance.
        self.assertLessEqual(abs(_effective_grid_area(points[:,0],points[:,1])-_effective_grid_area(dense[:,0],dense[:,1])),.25**2)

    def test_atc_area_calibration_policy(self):
        self.assertEqual(calibration_status(800,900),"calibrated")
        self.assertEqual(calibration_status(700,900),"outside_tolerance")

    def test_atc_failed_default_calibration_has_sweep(self):
        path=Path(__file__).resolve().parents[1]/"paper/stats/atc_area_calibration.json"
        if not path.exists():
            self.skipTest("ATC calibration artifact has not been generated")
        result=json.loads(path.read_text())
        # The "default" row must reflect the parameters analyze() actually uses,
        # not merely some swept setting, or a passing/failing calibration status
        # would misrepresent the production area computation.
        self.assertEqual(result["default"]["cell_size_m"],AREA_CELL_SIZE_M)
        self.assertEqual(result["default"]["observation_mass_fraction"],AREA_MASS_FRACTION)
        if result["status"]=="outside_tolerance":
            self.assertTrue(result["sweep"])
            self.assertLess(
                abs(result["closest_swept_setting"]["ratio_to_documented"]-1),
                abs(result["default"]["ratio_to_documented"]-1),
            )

    def test_calibrate_atc_default_matches_production_area_mass_fraction(self):
        # Regression test for A2-001: calibrate_atc's selected "default" row
        # must always be the exact (cell_size, mass_fraction) pair that
        # stats.py's analyze() uses in production, so the calibration report
        # never validates a parameterization that no reported density number
        # actually uses.
        rng=np.random.default_rng(0)
        x=rng.uniform(0,20,size=5_000)
        y=rng.uniform(0,20,size=5_000)
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir=Path(tmp)
            converted=dataset_dir/"converted"
            converted.mkdir(parents=True)
            _write_source_json(dataset_dir)
            table=pa.table({"x":pa.array(x,type=pa.float32()),"y":pa.array(y,type=pa.float32())})
            pq.write_table(table,converted/"rec1.traj.parquet")
            result=calibrate_atc(dataset_dir)
        self.assertEqual(result["default"]["cell_size_m"],AREA_CELL_SIZE_M)
        self.assertEqual(result["default"]["observation_mass_fraction"],AREA_MASS_FRACTION)

    def test_effective_grid_area_matches_the_mass_definition(self):
        # A2-011: the shared helper must reproduce the mass-area definition
        # (cell_size^2 x smallest cell set holding mass_fraction of observations)
        # on arbitrary counters, including empty ones.
        rng=np.random.default_rng(11)
        for _ in range(200):
            size=int(rng.integers(0,40))
            counter=Counter({int(k):int(v) for k,v in zip(rng.integers(-10**6,10**6,size),rng.integers(1,500,size))})
            for cell in (.25,.5,1.):
                for fraction in (0.,.25,.75,.95,1.):
                    area=_effective_grid_area_from_counts(counter,cell,fraction,min_cells=0)
                    if not counter:
                        self.assertTrue(math.isnan(area),"an empty counter has no defined area")
                    else:
                        self.assertEqual(area,cell*cell*_mass_cells_reference(list(counter.values()),fraction))

    def test_effective_grid_area_min_cells_zero_is_required_for_sparse_counters(self):
        # A2-011: area_calibration passes min_cells=0 deliberately. The production
        # default rejects counters occupying fewer than AREA_MIN_CELLS cells by
        # returning nan, so a drop-in call without min_cells=0 would silently turn
        # every sparse calibration row into nan instead of measuring it.
        sparse=Counter({1:5,2:3,3:1})
        occupied=_mass_cells_reference(list(sparse.values()),AREA_MASS_FRACTION)
        self.assertLess(occupied,AREA_MIN_CELLS,"fixture must exercise the rejection branch")
        self.assertEqual(_effective_grid_area_from_counts(sparse,.25,AREA_MASS_FRACTION,min_cells=0),.25*.25*occupied)
        self.assertTrue(math.isnan(_effective_grid_area_from_counts(sparse,.25,AREA_MASS_FRACTION)))

    def test_strided_step_occupancy_matches_the_sampling_grid(self):
        # A2-010: the vectorised occupancy must equal the range()-based grid it
        # stands for, in both values and length, for gaps, negative steps, single
        # steps, and spans that are not a multiple of the stride.
        cases=[([7],1),([7],3),
               ([-9,-9,-8,-5,-5,-5,-1,0,3],1),([-9,-9,-8,-5,-5,-5,-1,0,3],2),
               ([-40,-37,-36,-36,-33],4),
               ([0,1,2,50,51,52],1),([0,1,2,50,51,52],3),
               (list(range(10)),3),(list(range(11)),4),
               ([5,5,5,5],5),([0,1,2],10),([0,1,3,7],2)]
        rng=np.random.default_rng(2026)
        cases+=[(rng.integers(-500,500,int(rng.integers(1,60))).tolist(),int(rng.integers(1,9))) for _ in range(50)]
        for steps,stride in cases:
            unique,counts=np.unique(np.asarray(steps,dtype=np.int32),return_counts=True)
            occupancy=_strided_step_occupancy(unique,counts,stride)
            expected=_occupancy_reference(unique,counts,stride)
            message=f"steps={sorted(steps)} stride={stride}"
            self.assertEqual(occupancy.shape,expected.shape,message)
            self.assertEqual(occupancy.dtype,np.float64,message)
            self.assertTrue(np.array_equal(occupancy,expected),message)
            # Total agents are conserved for every step that lands on the grid.
            self.assertEqual(occupancy.sum(),counts[(unique-unique.min())%stride==0].sum(),message)

    def test_calibration_union_area_matches_direct_coarse_binning(self):
        # A2-014: coarsen() aggregates the 0.25 m grid by an integer factor. Because
        # floor(floor(x/a)/n)==floor(x/(a*n)) exactly for these power-of-two cell
        # sizes, the swept union area must equal binning the raw coordinates at the
        # coarse size directly -- an independent check of the vectorised coarsening
        # (and of the min_cells=0 call) that never re-runs coarsen's own arithmetic.
        rng=np.random.default_rng(0)
        x=rng.uniform(0,20,size=5_000);y=rng.uniform(0,20,size=5_000)
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir=Path(tmp);converted=dataset_dir/"converted";converted.mkdir(parents=True)
            _write_source_json(dataset_dir)
            table=pa.table({"x":pa.array(x,type=pa.float32()),"y":pa.array(y,type=pa.float32())})
            pq.write_table(table,converted/"rec1.traj.parquet")
            result=calibrate_atc(dataset_dir)
        stored=np.asarray(table.column("x")),np.asarray(table.column("y"))
        for row in result["sweep"]:
            size=row["cell_size_m"];fraction=row["observation_mass_fraction"]
            direct=_effective_grid_area_from_counts(_grid_cell_counts(*stored,size),size,fraction,min_cells=0)
            self.assertEqual(row["union_area_m2"],direct,f"cell_size={size} mass_fraction={fraction}")
            # One recording: the union and the median recording are the same set.
            self.assertEqual(row["median_recording_area_m2"],direct)

    def test_calibration_reports_finite_areas_for_a_sparse_recording(self):
        # A2-011 regression: calibrate_atc must pass min_cells=0 at BOTH the per-recording
        # and the union call site. A recording occupying fewer than AREA_MIN_CELLS cells
        # is measured, not rejected -- dropping min_cells=0 would silently turn the
        # published mean/median/ratio columns into nan. A dense recording alone cannot
        # detect that, because it always clears the production threshold.
        rng=np.random.default_rng(0)
        dense=rng.uniform(0,20,size=(2,5_000))
        # 40 points inside one square metre: at most 16 occupied 0.25 m cells, so the
        # production AREA_MIN_CELLS rejection would fire at every swept cell size.
        sparse=rng.uniform(30,31,size=(2,40))
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir=Path(tmp);converted=dataset_dir/"converted";converted.mkdir(parents=True)
            _write_source_json(dataset_dir)
            tables=[]
            for name,(px,py) in (("rec1",dense),("rec2",sparse)):
                table=pa.table({"x":pa.array(px,type=pa.float32()),"y":pa.array(py,type=pa.float32())})
                pq.write_table(table,converted/f"{name}.traj.parquet");tables.append(table)
            result=calibrate_atc(dataset_dir)
        stored=[(np.asarray(table.column("x")),np.asarray(table.column("y"))) for table in tables]
        combined=(np.concatenate([x for x,_ in stored]),np.concatenate([y for _,y in stored]))
        self.assertEqual(len(result["sweep"]),15)
        for row in result["sweep"]:
            size=row["cell_size_m"];fraction=row["observation_mass_fraction"]
            message=f"cell_size={size} mass_fraction={fraction}"
            counts=[_grid_cell_counts(x,y,size) for x,y in stored]
            occupied=_mass_cells_reference(list(counts[1].values()),fraction)
            self.assertLess(occupied,AREA_MIN_CELLS,f"sparse fixture must stay below the rejection threshold: {message}")
            areas=[size*size*_mass_cells_reference(list(counter.values()),fraction) for counter in counts]
            for column in ("mean_recording_area_m2","median_recording_area_m2","ratio_to_documented"):
                self.assertTrue(math.isfinite(row[column]),f"{column} must be measured, not rejected: {message}")
            # Two recordings, so numpy's median is their mean: both columns are pinned to
            # the same independently computed pair of mass areas.
            self.assertAlmostEqual(row["mean_recording_area_m2"],sum(areas)/2,places=9,msg=message)
            self.assertAlmostEqual(row["median_recording_area_m2"],sum(areas)/2,places=9,msg=message)
            self.assertAlmostEqual(row["ratio_to_documented"],sum(areas)/2/900.0,places=9,msg=message)
            union=size*size*_mass_cells_reference(list(_grid_cell_counts(*combined,size).values()),fraction)
            self.assertEqual(row["union_area_m2"],union,message)

    def test_calibrate_atc_documented_area_tracks_source_json(self):
        # Q-006: area_calibration.py used to divide by the literal 900.0 and
        # hardcode "dataset": "atc" no matter what source.json said or what the
        # directory was named. Reading area_m2/area_source/dataset name from
        # source.json (the same file stats.analyze() reads via metadata_path)
        # means the artifact and analyze()'s documented_area_m2 cannot disagree
        # about the same dataset. This must fail against the pre-fix code,
        # which ignores source.json entirely and always returns 900.0/"atc".
        rng=np.random.default_rng(0)
        x=rng.uniform(0,20,size=500);y=rng.uniform(0,20,size=500)
        table=pa.table({"x":pa.array(x,type=pa.float32()),"y":pa.array(y,type=pa.float32())})
        for area_m2 in (900.0,1234.5):
            with tempfile.TemporaryDirectory() as tmp:
                dataset_dir=Path(tmp)/"some_other_dataset";converted=dataset_dir/"converted";converted.mkdir(parents=True)
                _write_source_json(dataset_dir,area_m2=area_m2,area_source=f"fixture area {area_m2}")
                pq.write_table(table,converted/"rec1.traj.parquet")
                result=calibrate_atc(dataset_dir)
            self.assertEqual(result["documented_area_m2"],area_m2)
            self.assertEqual(result["documented_area_source"],f"fixture area {area_m2}")
            # No "atc" anywhere in the directory name: the label must come from
            # dataset_dir.name, not a hardcoded "atc".
            self.assertEqual(result["dataset"],"some_other_dataset")
            for row in result["sweep"]:
                self.assertAlmostEqual(row["ratio_to_documented"]*area_m2,row["median_recording_area_m2"],places=9)

    def test_calibrate_atc_requires_a_documented_area(self):
        # Q-006: calibrate_atc must not silently fall back to a stale constant
        # when source.json is missing or lacks area_m2 -- that is exactly the
        # bug being fixed, so the absence of a documented area must raise.
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir=Path(tmp)/"no_source_json";converted=dataset_dir/"converted";converted.mkdir(parents=True)
            table=pa.table({"x":pa.array([0.],type=pa.float32()),"y":pa.array([0.],type=pa.float32())})
            pq.write_table(table,converted/"rec1.traj.parquet")
            with self.assertRaises(Exception):
                calibrate_atc(dataset_dir)

    def test_linear_track_statistics(self):
        xy=np.c_[np.arange(20,dtype=float),np.zeros(20)]
        self.assertAlmostEqual(cv_fde(xy),0.0,places=6)
        self.assertAlmostEqual(path_efficiency(xy),1.0)
        self.assertAlmostEqual(heading_entropy(xy),0.0)
        self.assertAlmostEqual(rms_jerk(xy),0.0)

    def test_cv_uses_only_the_final_observed_displacement(self):
        xy=np.zeros((20,2),dtype=float)
        xy[:8,0]=[0,0,0,0,0,0,0,1]
        xy[8:,0]=1+np.arange(1,13)
        self.assertAlmostEqual(cv_fde(xy),0.0,places=12)

    def test_min_time_to_closest_approach_scene_head_on_pair(self):
        # Primary at the origin heading +x at 1 m/s, neighbour 10 m ahead
        # heading -x at 1 m/s: the 10 m gap closes at 2 m/s, so the two agents
        # coincide 10/2 = 5.0 s after the last observed step.
        #
        # Definition pinned here: min_time_to_closest_approach_scene returns the
        # time of *minimum separation*, gated on that separation being under the
        # 2*radius=0.66 m contact threshold (radius=0.33 m, OpenTraj's convention;
        # see MIN_TCA_CONTACT_RADIUS_M / K-001). That is later than the time the
        # gap first reaches contact, which for this geometry would be
        # (10-0.66)/2 = 4.67 s.
        xy,valid=_linear_scene([(0.,0.),(10.,0.)],[(1.,0.),(-1.,0.)])
        self.assertAlmostEqual(min_time_to_closest_approach_scene(xy,valid),5.0,places=9)

    def test_min_time_to_closest_approach_scene_reports_the_earliest_neighbour(self):
        # Two head-on neighbours 10 m and 4 m ahead, each closing at 2 m/s,
        # coincide with the primary at 5.0 s and 2.0 s; the scene statistic is
        # the earlier of the two.
        xy,valid=_linear_scene([(0.,0.),(10.,0.),(4.,0.)],[(1.,0.),(-1.,0.),(-1.,0.)])
        self.assertAlmostEqual(min_time_to_closest_approach_scene(xy,valid),2.0,places=9)

    def test_min_time_to_closest_approach_scene_contact_threshold_is_two_radii(self):
        # Two agents of radius 0.33 m touch when their centres are 0.66 m apart,
        # so head-on courses offset by 0.4 m in y (< 0.66 m) still count as a
        # collision; closest approach is again the 10 m gap closing at 2 m/s,
        # i.e. 5.0 s.
        xy,valid=_linear_scene([(0.,0.),(10.,.4)],[(1.,0.),(-1.,0.)])
        self.assertAlmostEqual(min_time_to_closest_approach_scene(xy,valid),5.0,places=9)

    def test_min_time_to_closest_approach_scene_ignores_a_wide_miss(self):
        # Same head-on courses offset by 2 m in y: the extrapolated tracks are
        # parallel lines 2 m apart, so the minimum separation is 2.0 m, far
        # above the 2*radius=0.66 m contact threshold.
        xy,valid=_linear_scene([(0.,0.),(10.,2.)],[(1.,0.),(-1.,0.)])
        self.assertEqual(min_time_to_closest_approach_scene(xy,valid),float("inf"))

    def test_min_time_to_closest_approach_scene_ignores_a_receding_neighbour(self):
        # The neighbour is already 10 m ahead and pulling away at 1 m/s, so the
        # separation grows monotonically and closest approach lies in the past.
        xy,valid=_linear_scene([(0.,0.),(10.,0.)],[(1.,0.),(2.,0.)])
        self.assertEqual(min_time_to_closest_approach_scene(xy,valid),float("inf"))

    def test_min_time_to_closest_approach_scene_degenerate_scenes(self):
        xy,valid=_linear_scene([(0.,0.)],[(1.,0.)])
        self.assertEqual(min_time_to_closest_approach_scene(xy,valid),float("inf"),"a scene with no neighbour has no time-to-closest-approach")
        # A neighbour co-moving with the primary has zero relative velocity, so
        # the separation never changes and no closest-approach time exists.
        xy,valid=_linear_scene([(0.,0.),(10.,0.)],[(1.,0.),(1.,0.)])
        self.assertEqual(min_time_to_closest_approach_scene(xy,valid),float("inf"))
        # A neighbour needs both of the final two observed steps to define its
        # velocity; dropping either one makes it unusable.
        xy,valid=_linear_scene([(0.,0.),(10.,0.)],[(1.,0.),(-1.,0.)])
        for step in (6,7):  # obs-2 and obs-1 for the default obs=8
            masked=valid.copy();masked[step,1]=False
            self.assertEqual(min_time_to_closest_approach_scene(xy,masked),float("inf"),f"neighbour invalid at step {step}")


if __name__=="__main__":unittest.main()
