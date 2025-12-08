import os
import sys

# Setup OpenSim Environment
opensim_root = r'C:\OpenSim 4.5'
opensim_bin_path = os.path.join(opensim_root, 'bin')
opensim_sdk_python_path = os.path.join(opensim_root, 'sdk', 'Python')

# 1. Add bin to PATH/DLL directory
if os.path.exists(opensim_bin_path):
    try:
        os.add_dll_directory(opensim_bin_path)
    except AttributeError:
        os.environ['PATH'] = opensim_bin_path + ';' + os.environ['PATH']
else:
    print(f"Warning: OpenSim bin path {opensim_bin_path} not found.")

# 2. Add SDK/Python to sys.path
if os.path.exists(opensim_sdk_python_path):
    sys.path.append(opensim_sdk_python_path)
    print(f"Added {opensim_sdk_python_path} to sys.path")
else:
    # Try finding it in site-packages if installed via pip
    pass

import opensim as osim
import numpy as np
import time
import csv
import itertools

def get_influenced_coordinates(model, muscle):
    """
    Identifies which coordinates affect the length of the given muscle.
    """
    state = model.initSystem()
    coords = model.getCoordinateSet()
    influenced_coords = []
    
    initial_length = muscle.getLength(state)
    
    for i in range(coords.getSize()):
        coord = coords.get(i)
        if coord.getMotionType() == osim.Coordinate.Coupled:
            continue
            
        # Perturb coordinate
        orig_val = coord.getValue(state)
        range_min = coord.getRangeMin()
        range_max = coord.getRangeMax()
        
        # Test distinct value
        test_val = range_min + 0.5 * (range_max - range_min)
        if abs(test_val - orig_val) < 1e-4:
            test_val = range_min + 0.8 * (range_max - range_min)
            
        coord.setValue(state, test_val)
        new_length = muscle.getLength(state)
        
        if abs(new_length - initial_length) > 1e-5:
            influenced_coords.append(coord)
            
        # Reset
        coord.setValue(state, orig_val)
        
    return influenced_coords

def estimate_mtu_length_range(model, muscle, num_samples=5):
    """
    Estimates the min and max MTU length by sampling the joint space of influenced coordinates.
    Referenced from myoconverter's getMuscleLengthList.
    """
    coords = get_influenced_coordinates(model, muscle)
    if not coords:
        print(f"Warning: No coordinates found that influence {muscle.getName()}. Using default range.")
        l_opt = muscle.getOptimalFiberLength()
        return 0.5 * l_opt, 1.5 * l_opt
        
    print(f"Muscle {muscle.getName()} is influenced by: {[c.getName() for c in coords]}")
    
    state = model.initSystem()
    
    # Generate mesh for each coordinate
    coord_ranges = []
    for coord in coords:
        rmin = coord.getRangeMin()
        rmax = coord.getRangeMax()
        # Avoid exact limits in case of numerical issues
        coord_ranges.append(np.linspace(rmin + 1e-3, rmax - 1e-3, num_samples))
        
    min_l = float('inf')
    max_l = float('-inf')
    
    # Iterate through all combinations
    for p in itertools.product(*coord_ranges):
        for i, val in enumerate(p):
            coords[i].setValue(state, val)
            
        l_mtu = muscle.getLength(state)
        if l_mtu < min_l: min_l = l_mtu
        if l_mtu > max_l: max_l = l_mtu
        
    # Add a small margin
    margin = 0.05 * (max_l - min_l)
    return max(0.01, min_l - margin), max_l + margin

def create_virtual_experiment_model(ref_muscle):
    """
    Creates a simple 1-DOF model (Slider) and attaches a copy of the reference muscle.
    The slider coordinate corresponds directly to MTU length.
    """
    model = osim.Model()
    model.setName("VirtualMuscleTester")
    
    # 1. Add a massive block to pull
    ground = model.getGround()
    block = osim.Body("block", 100.0, osim.Vec3(0), osim.Inertia(1,1,1,0,0,0))
    model.addBody(block)
    
    # 2. Add Slider Joint (q = MTU length)
    slider = osim.SliderJoint("slider", ground, block)
    coord = slider.updCoordinate()
    coord.setName("mtu_length")
    # Translate along X axis
    # In OpenSim 4.x, SliderJoint automatically has a Translation coordinate
    
    model.addJoint(slider)
    
    # 3. Clone and Add Muscle
    try:
        new_muscle = ref_muscle.clone()
    except:
        new_muscle = osim.Millard2012EquilibriumMuscle.safeDownCast(ref_muscle).clone()
        
    new_muscle.setName("test_muscle")
    
    # Update Geometry Path: Ground(0,0,0) -> Block(0,0,0)
    # The SliderJoint translates Block along X axis.
    # So if q=0.5, Block is at (0.5, 0, 0).
    # Path length = sqrt((0.5-0)^2 + ...) = 0.5.
    # PERFECT.
    
    path = new_muscle.updGeometryPath()
    # path.clearPath() # Not available in Python bindings sometimes
    path.updPathPointSet().clearAndDestroy()
    path.updWrapSet().clearAndDestroy()
    
    # Use appendNewPathPoint which is safer and easier
    path.appendNewPathPoint("origin", ground, osim.Vec3(0,0,0))
    path.appendNewPathPoint("insertion", block, osim.Vec3(0,0,0))
    
    # p1 = osim.PathPoint()
    # p1.setName("origin")
    # p1.setParentFrame(ground)
    # p1.setLocation(osim.Vec3(0,0,0))
    # 
    # p2 = osim.PathPoint()
    # p2.setName("insertion")
    # p2.setParentFrame(block)
    # p2.setLocation(osim.Vec3(0,0,0))
    # 
    # path.addPathPoint(p1)
    # path.addPathPoint(p2)
    
    # model.addMuscle(new_muscle) # addMuscle is deprecated or removed in newer versions
    model.addForce(new_muscle)
    
    # Needs to finalize connections before initSystem
    model.finalizeConnections()
    
    # Downcast to Millard2012EquilibriumMuscle to ensure we access all methods
    # We know the original was Millard.
    sim_muscle_downcast = osim.Millard2012EquilibriumMuscle.safeDownCast(new_muscle)
    if sim_muscle_downcast is not None:
        # slightly higher damping for stability in velocity solves
        sim_muscle_downcast.setFiberDamping(0.2)
    
    return model, sim_muscle_downcast

def run_velocity_test(muscle_ref, output_dir="osim_muscle_data_sim", norm_velocities=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    muscle_name = muscle_ref.getName()
    print(f"Starting simulation for {muscle_name}...")
    
    # Set MTU length range from l_opt / l_slack
    l_opt = muscle_ref.getOptimalFiberLength()
    l_slack = muscle_ref.getTendonSlackLength()
    min_len = l_slack
    max_len = 2.0 * l_opt + l_slack
    print(f"[{muscle_name}] l_opt={l_opt:.6f}, l_slack={l_slack:.6f}")
    print(f"[{muscle_name}] MTU range formula: min_len = l_slack -> {min_len:.6f}, max_len = 2.0*l_opt + l_slack -> {max_len:.6f}")

    # Define ranges
    num_points = 40 # finer resolution for smoother curve
    mtu_lengths = np.linspace(min_len, max_len, num_points)
    
    # Velocities to test (only v=0 unless overridden)
    if norm_velocities is None:
        norm_velocities = np.array([0.0])
    v_max = muscle_ref.getMaxContractionVelocity()
    
    results = {
        'mtu_lengths': mtu_lengths,
        'norm_velocities': norm_velocities,
        'active_force': np.zeros((len(mtu_lengths), len(norm_velocities))),
        'passive_force': np.zeros((len(mtu_lengths), len(norm_velocities))),
        'total_force': np.zeros((len(mtu_lengths), len(norm_velocities)))
    }
    
    # Create Virtual Model
    model, sim_muscle = create_virtual_experiment_model(muscle_ref)
    
    # Access coordinate
    # SliderJoint name is "slider", coordinate name is usually "slider_coord_0" or similar unless renamed
    # We renamed it to "mtu_length"
    coord = model.updCoordinateSet().get("mtu_length")
    coord.setPrescribedFunction(osim.Constant(0.0)) # Default
    coord.setDefaultLocked(False) 
    
    # Initialize System
    state = model.initSystem()

    def eval_force_with_dynamics(st, activation_level):
        sim_muscle.setActivation(st, activation_level)
        # realize to velocity so muscle velocity is considered
        model.realizeVelocity(st)
        # try the public accessor; fall back to equilibrium if unavailable
        try:
            info = sim_muscle.getMuscleDynamicsInfo(st)
            try:
                return info.tendonForce
            except Exception:
                try:
                    return info.fiberForceAlongTendon
                except Exception:
                    pass
        except Exception:
            pass
        # fallback: equilibrium + fiber force along tendon
        try:
            sim_muscle.computeEquilibrium(st)
            return sim_muscle.getFiberForceAlongTendon(st)
        except Exception:
            return np.nan

    for j, v_norm in enumerate(norm_velocities):
        v_target = v_norm * v_max
        print(f"  Simulating v_norm={v_norm:.2f} ({v_target:.3f} m/s)")
        coord.setPrescribedFunction(osim.Constant(0.0))
        coord.setDefaultLocked(False)
        coord.setLocked(state, False)
        for i, l in enumerate(mtu_lengths):
            coord.setValue(state, l)
            coord.setSpeedValue(state, v_target)
            
            # Passive at this length/velocity (dynamics info)
            f_passive = eval_force_with_dynamics(state, 0.0)
            
            # Active at this length/velocity (dynamics info)
            f_total = eval_force_with_dynamics(state, 1.0)
            
            results['passive_force'][i, j] = f_passive
            results['total_force'][i, j] = f_total
            results['active_force'][i, j] = f_total - f_passive
    # Save results to CSV (Using csv module instead of pandas)
    def write_surface(name, data):
        fname = os.path.join(output_dir, f"{muscle_name}_sim_{name}.csv")
        with open(fname, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            header = ['mtu_length'] + [f"{v:.5f}" for v in norm_velocities]
            writer.writerow(header)
            for i, l_mtu in enumerate(mtu_lengths):
                row = [f"{l_mtu:.8f}"]
                for j in range(len(norm_velocities)):
                    row.append(f"{data[i, j]:.8f}")
                writer.writerow(row)
        return fname

    f_active = write_surface("active", results['active_force'])
    f_passive = write_surface("passive", results['passive_force'])
    f_total = write_surface("total", results['total_force'])
    
    print(f"Finished {muscle_name} simulation.")
    print(f"Saved active surface: {f_active}")
    print(f"Saved passive surface: {f_passive}")
    print(f"Saved total surface: {f_total}")
    return results

if __name__ == "__main__":
    model_path = 'opensim_models/Rajagopal/RajagopalLaiUhlrich2023.osim'
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
    else:
        full_model = osim.Model(model_path)
        muscles = full_model.getMuscles()
        out_dir = "osim_muscle_data_sim"
        try:
            import matplotlib.pyplot as plt
        except Exception as e:
            plt = None
            print(f"Plot import failed: {e}")

        summary_curves = []
        for idx in range(muscles.getSize()):
            muscle = muscles.get(idx)
            mname = muscle.getName()
            if mname.endswith("_l"):
                print(f"Skipping left-side muscle {mname}")
                continue
            sim_res = run_velocity_test(muscle, output_dir=out_dir, norm_velocities=np.array([0.0]))
            summary_curves.append((
                mname,
                sim_res['mtu_lengths'],
                sim_res['active_force'][:, 0],
                sim_res['passive_force'][:, 0],
                sim_res['total_force'][:, 0],
            ))
            if plt is not None:
                try:
                    plt.figure(figsize=(8,5))
                    plt.plot(sim_res['mtu_lengths'], sim_res['active_force'][:, 0], label="active (v=0)")
                    plt.plot(sim_res['mtu_lengths'], sim_res['passive_force'][:, 0], label="passive (v=0)")
                    plt.plot(sim_res['mtu_lengths'], sim_res['total_force'][:, 0], label="total (v=0)")
                    plt.xlabel("MTU length (m)")
                    plt.ylabel("Force (N)")
                    plt.title(f"{mname} length-force (v=0)")
                    plt.legend()
                    plt.tight_layout()
                    plot_path = os.path.join(out_dir, f"{mname}_lf_v0.png")
                    plt.savefig(plot_path, dpi=200)
                    plt.close()
                    print(f"Saved plot: {plot_path}")
                except Exception as e:
                    print(f"Plotting failed for {mname}: {e}")

        # combined plot: all muscles active (v=0) on one figure
        if plt is not None and summary_curves:
            try:
                plt.figure(figsize=(10,8))
                cmap = plt.get_cmap("tab20")
                for i, (mname, lengths, active_force, _, _) in enumerate(summary_curves):
                    plt.plot(lengths, active_force, label=mname, color=cmap(i % 20), linewidth=1.2)
                plt.xlabel("MTU length (m)")
                plt.ylabel("Active force (v=0) (N)")
                plt.title("All muscles length-force (v=0, right side only)")
                plt.legend(fontsize="x-small", ncol=2)
                plt.tight_layout()
                combo_path = os.path.join(out_dir, "all_muscles_length_force_v0.png")
                plt.savefig(combo_path, dpi=200)
                plt.close()
                print(f"Saved combined plot: {combo_path}")
            except Exception as e:
                print(f"Combined plotting failed: {e}")

        # grid plot: multiple subplots (rows/cols) each muscle active v=0
        if plt is not None and summary_curves:
            try:
                n = len(summary_curves)
                ncols = 4
                nrows = int(np.ceil(n / ncols))
                fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), squeeze=False)
                cmap = plt.get_cmap("tab20")
                for idx, (mname, lengths, active_force, passive_force, total_force) in enumerate(summary_curves):
                    r = idx // ncols
                    c = idx % ncols
                    ax = axes[r][c]
                    ax.plot(lengths, active_force, color=cmap(idx % 20), linewidth=1.3, label=f"{mname} active")
                    ax.plot(lengths, passive_force, color="gray", linewidth=1.0, linestyle="--", label="passive")
                    ax.plot(lengths, total_force, color="black", linewidth=1.0, linestyle=":", label="total")
                    ax.set_title(mname)
                    ax.set_xlabel("MTU length (m)")
                    ax.set_ylabel("Active force (v=0) (N)")
                    ax.grid(True, alpha=0.3)
                    ax.legend(fontsize="x-small")
                # remove unused axes
                for k in range(len(summary_curves), nrows*ncols):
                    r = k // ncols
                    c = k % ncols
                    fig.delaxes(axes[r][c])
                fig.tight_layout()
                grid_path = os.path.join(out_dir, "all_muscles_length_force_v0_grid.png")
                plt.savefig(grid_path, dpi=200)
                plt.close(fig)
                print(f"Saved grid plot: {grid_path}")
            except Exception as e:
                print(f"Grid plotting failed: {e}")
