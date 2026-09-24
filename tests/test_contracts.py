"""Research invariants: power conservation, repeatability, and RC identity."""
import unittest
import numpy as np
from hybridchannel.hybrid_channel import generate_full_channel
from hybridchannel.consistent_random_clusters import freeze_random_cluster_state,evaluate_random_clusters
from compare_baseline import rt_fixture


class Contracts(unittest.TestCase):
    def test_cluster_ray_power_conservation(self):
        for m in (1,3,20):
            out=generate_full_channel(rt_fixture(),'NTN-DenseUrban-LOS',np.zeros((2,3)),
                np.zeros((3,3)),fc_GHz=28,elev_deg=40,rng=np.random.default_rng(123),
                config={'M':m},tx_rx_distance_3D=900000)
            p=sum(c['power'] for c in out['hybrid']['clusters'])
            q=sum(r['power'] for r in out['rays'])
            self.assertAlmostEqual(p/q,1.,places=12)
            self.assertTrue(np.all(np.isfinite(out['channel']['H'])))
            # Each antenna pair carries the same sum of per-ray powers.
            # Accumulate in float64: summing many float32 powers otherwise
            # measures reduction round-off as well as coefficient precision.
            h128=out['channel']['H'].astype(np.complex128)
            np.testing.assert_allclose(np.sum(abs(h128)**2,axis=(2,3)),p,rtol=2e-7,atol=0)

    def test_frozen_rc_is_deterministic(self):
        state=freeze_random_cluster_state(rt_fixture(),'NTN-DenseUrban-LOS',[0,0,2],
            [715050,0,600000],28,elev_deg=40,seed=123)
        kw=dict(p_sat=[715050,75.6,600000],rt_paths_current=rt_fixture(),
                rx_array_geom=np.zeros((1,3)),tx_array_geom=np.zeros((1,3)))
        a=evaluate_random_clusters(state,**kw);b=evaluate_random_clusters(state,**kw)
        self.assertGreater(len(a['rays']),0)
        np.testing.assert_array_equal(a['channel']['H'],b['channel']['H'])
        np.testing.assert_array_equal(a['channel']['tau'],b['channel']['tau'])
        self.assertTrue(all(r['source']=='RC' for r in a['rays']))


if __name__=='__main__':unittest.main()
