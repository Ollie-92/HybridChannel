"""Compare the adapter with independent native OpenNTN calls; no RT tracing.

Reference flow: official example/channel_response.ipynb cells 6, 8, 10, 15, 27.
Run from the repository root: python hybrid_simulator/validate_openntn_mimo.py
"""
from pathlib import Path
import ast, copy, csv, hashlib, json, os, sys
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT / "third_party/openntn"))
from OpenNTN import Antenna, AntennaArray, DenseUrban, Urban, SubUrban
from sionna.phy import config as native_config
from sionna.phy.channel import cir_to_ofdm_channel
from hybrid_simulator.config import notebook_config, notebook_devices, validate_config
from hybrid_simulator.channel_generator import generate_channel
from hybrid_simulator.result import ChannelResult

OUT = ROOT / "results/validation/openntn_mimo"
OUT.mkdir(parents=True, exist_ok=True)
upstream = json.loads((ROOT / "third_party/openntn/UPSTREAM.json").read_text())
antenna_source = ROOT / "third_party/openntn/OpenNTN/antenna.py"
record = next(x for x in upstream["files"] if x["path"] == "OpenNTN/antenna.py")
assert hashlib.sha256(antenna_source.read_bytes()).hexdigest() == record["sha256"]

notebook = json.loads((ROOT.parent / "hybrid_simulator.ipynb").read_text())
configuration = next("".join(c["source"]) for c in notebook["cells"]
                     if c["cell_type"] == "code" and "settings = {" in "".join(c["source"]))
tree = ast.parse(configuration)
assignment = next(n for n in tree.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "settings" for t in n.targets))
scope = {"PROJECT": ROOT}
exec(compile(ast.Module(body=[assignment], type_ignores=[]), "notebook_settings", "exec"), scope)
base = scope["settings"]


def native_reference(c, legacy_single=False):
    """Direct upstream example flow, without adapter array or channel helpers."""
    native_config.seed = c["seed"]
    tf.random.set_seed(c["seed"])
    antennas = {}
    for name in ["ue", "satellite"]:
        spec = c["devices"][name]["antenna"]
        common = dict(polarization="dual" if spec["polarization"] == "VH" else "single",
                      polarization_type=spec["polarization"], antenna_pattern="omni",
                      carrier_frequency=c["radio"]["frequency_hz"], precision="single")
        if legacy_single:
            antennas[name] = Antenna(**common)
        else:
            antennas[name] = AntennaArray(num_rows=spec["num_rows"], num_cols=spec["num_cols"],
                vertical_spacing=spec["vertical_spacing"], horizontal_spacing=spec["horizontal_spacing"], **common)
    ut = np.array(c["devices"]["ue"]["position"])
    bs = np.array(c["devices"]["satellite"]["position"])
    delta = bs-ut
    elevation = float(np.rad2deg(np.arctan2(delta[2], np.linalg.norm(delta[:2]))))
    options = c["openntn"]
    cls = {"dense_urban": DenseUrban, "urban": Urban, "suburban": SubUrban}[options["scenario"]]
    model = cls(carrier_frequency=c["radio"]["frequency_hz"], ut_array=antennas["ue"],
                bs_array=antennas["satellite"], direction=c["direction"], elevation_angle=elevation,
                enable_pathloss=options["enable_pathloss"], enable_shadow_fading=options["enable_shadow_fading"],
                doppler_enabled=options["doppler_enabled"], precision="single")
    model.set_topology(ut_loc=tf.constant([[ut]], tf.float32), bs_loc=tf.constant([[bs]], tf.float32),
        ut_orientations=tf.constant([[np.deg2rad(c["devices"]["ue"]["orientation_deg"])]], tf.float32),
        bs_orientations=tf.constant([[np.deg2rad(c["devices"]["satellite"]["orientation_deg"])]], tf.float32),
        ut_velocities=tf.constant([[options["ue_velocity_m_s"]]], tf.float32),
        in_state=tf.constant([[False]]), los=options["los"])
    a, tau = model(num_time_samples=options["num_time_samples"], sampling_frequency=options["sampling_frequency_hz"])
    frequencies = np.linspace(-c["radio"]["bandwidth_hz"]/2, c["radio"]["bandwidth_hz"]/2, c["radio"]["frequency_points"])
    h = cir_to_ofdm_channel(tf.constant(frequencies, tf.float32), a, tau, normalize=False)
    return a.numpy(), tau.numpy(), h.numpy(), antennas


def compare(name, actual, expected):
    assert actual.shape == expected.shape, (name, actual.shape, expected.shape)
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    np.testing.assert_array_equal(actual, expected, err_msg=name)
    return float(np.max(np.abs(actual-expected), initial=0))


reports = []
cases = [
    ("single_v", (1,1,"V"), (1,1,"V"), "dense_urban", True, False),
    ("single_vh", (1,1,"VH"), (1,1,"VH"), "dense_urban", True, False),
    ("mimo_v", (1,4,"V"), (1,2,"V"), "dense_urban", True, True),
    ("mimo_vh_nlos", (2,2,"VH"), (1,2,"VH"), "dense_urban", False, True),
    ("mixed_h_vh", (2,3,"VH"), (2,1,"H"), "urban", True, True),
    ("suburban_nlos", (1,3,"H"), (2,2,"V"), "suburban", False, False),
]
for case_id, sat_spec, ue_spec, scenario, los, doppler in cases:
    for direction in ["downlink", "uplink"]:
        settings = copy.deepcopy(base)
        settings.update(mode="3gpp", topology="fixed", directions=[direction], frequency_hz=2e9, seed=12345)
        for name, endpoint in settings["endpoints"].items():
            endpoint["enabled"] = name in ["sat1", "ue1"]
        for name,spec in [("sat1",sat_spec),("ue1",ue_spec)]:
            settings["endpoints"][name]["antenna"].update(num_rows=spec[0], num_cols=spec[1], polarization=spec[2],
                vertical_spacing=.7, horizontal_spacing=.4)
            settings["endpoints"][name]["orientation_deg"]=[17.,-12.,8.]
        settings["endpoints"]["sat1"]["position_m"]=[600000.,0.,600000.]
        settings["endpoints"]["ue1"]["position_m"]=[0.,0.,2.]
        settings["openntn"].update(scenario=scenario, los=los, num_time_samples=4,
            sampling_frequency_hz=10000., ue_velocity_m_s=[3.,1.,0.], doppler_enabled=doppler)
        c = notebook_config(settings)
        devices = notebook_devices(settings)
        c["devices"]={"satellite":devices["sat1"],"ue":devices["ue1"]}
        c["radio"].update(bandwidth_hz=1e6,frequency_points=5)
        c=validate_config(c)
        a,tau,cfr,antennas=native_reference(c,legacy_single=case_id.startswith('single'))
        result=generate_channel(c)
        errors={
            'native_a':compare('native CIR',result.extra_arrays['openntn_a'],a),
            'native_tau':compare('native delays',result.extra_arrays['openntn_tau'],tau),
            'native_cfr':compare('native CFR',result.extra_arrays['openntn_cfr'],cfr),
        }
        tx_name,rx_name=('satellite','ue') if direction=='downlink' else ('ue','satellite')
        orders={}
        for name in ['satellite','ue']:
            ant=antennas[name]
            # Independently derive the port permutation from native polarization indices.
            p1=np.asarray(ant.ant_ind_pol1)
            p2=np.asarray(ant.ant_ind_pol2) if c['devices'][name]['polarization']=='VH' else np.array([],dtype=int)
            orders[name]=np.stack([p1,p2],axis=1).ravel() if len(p2) else p1
        expected_a=a[0,0,:,0,:,:,:][orders[rx_name]][:,orders[tx_name]]
        expected_cfr=cfr[0,0,:,0,:,:,:][orders[rx_name]][:,orders[tx_name]]
        errors['output_a']=compare('mapped CIR',result.a,expected_a)
        errors['output_cfr']=compare('mapped CFR',result.extra_arrays['cfr_time'],expected_cfr)
        errors['matrix']=compare('coherent matrix',result.h,expected_a.sum(axis=2))
        for name in ['satellite','ue']:
            # Verify shared notebook array geometry against rotated native element positions.
            local=antennas[name].ant_pos_pol1.numpy()
            # Independent Rz(alpha) Ry(beta) Rx(gamma) rotation convention.
            alpha,beta,gamma=np.deg2rad(c['devices'][name]['orientation_deg'])
            rz=np.array([[np.cos(alpha),-np.sin(alpha),0],[np.sin(alpha),np.cos(alpha),0],[0,0,1]])
            ry=np.array([[np.cos(beta),0,np.sin(beta)],[0,1,0],[-np.sin(beta),0,np.cos(beta)]])
            rx=np.array([[1,0,0],[0,np.cos(gamma),-np.sin(gamma)],[0,np.sin(gamma),np.cos(gamma)]])
            np.testing.assert_allclose(local @ (rz@ry@rx).T,c['devices'][name]['array_offsets_m'],rtol=2e-6,atol=1e-8)
        folder=OUT/(case_id+'_'+direction)
        result.save(folder)
        np.savez_compressed(folder/'native_reference.npz',a=a,tau=tau,cfr=cfr)
        (folder/'config.json').write_text(json.dumps(c,indent=2))
        reloaded=ChannelResult.load(folder);compare('NPZ roundtrip',reloaded.a,expected_a)
        report=dict(case=case_id,direction=direction,scenario=scenario,los=los,shape=list(result.a.shape),max_absolute_errors=errors,status='pass')
        reports.append(report)
        print(json.dumps(report),flush=True)
        (OUT/'validation.json').write_text(json.dumps({'upstream_commit':upstream['commit'],'cases':reports},indent=2))
with (OUT/'validation.csv').open('w',newline='',encoding='utf-8-sig') as f:
    writer=csv.writer(f);writer.writerow(['case','direction','scenario','los','shape','max_abs_error','status'])
    for r in reports:writer.writerow([r['case'],r['direction'],r['scenario'],r['los'],r['shape'],max(r['max_absolute_errors'].values()),r['status']])
print('PASS:',len(reports),'native comparisons',flush=True)
