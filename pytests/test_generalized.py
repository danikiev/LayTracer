
import pytest
import numpy as np
import pandas as pd
from laytracer import trace_rays, transmission_normal

# Simple 2-layer model
# Layer 0: Vp=2000, Vs=1000, Rho=2000, Depth=0
# Layer 1: Vp=4000, Vs=2000, Rho=2500, Depth=1000
test_model = pd.DataFrame({
    "Depth": [0.0, 1000.0],
    "Vp": [2000.0, 4000.0],
    "Vs": [1000.0, 2000.0],
    "Rho": [2000.0, 2500.0],
    "Qp": [500.0, 1000.0],
    "Qs": [250.0, 500.0],
})

def test_pp_reflection_vertical():
    """Test vertical P-P reflection: 0 -> 1000 -> 0"""
    # Distance = 2000m
    # Vp = 2000 m/s
    # Time = 1.0 s
    
    res = trace_rays(
        sources=[0,0,0],
        receivers=[0,0,0],
        velocity_df=test_model,
        source_phase="P",
        reflection=[(1000.0, "P")]
    )
    
    assert res.travel_times is not None
    assert np.isclose(res.travel_times[0], 1.0, rtol=1e-5)
    assert np.isclose(res.ray_parameters[0], 0.0, atol=1e-9)

def test_ps_reflection_vertical():
    """Test vertical P-S reflection: P down (0->1000), S up (1000->0)"""
    # Down: 1000m / 2000m/s = 0.5s
    # Up: 1000m / 1000m/s = 1.0s
    # Total = 1.5s
    
    res = trace_rays(
        sources=[0,0,0],
        receivers=[0,0,0],
        velocity_df=test_model,
        source_phase="P",
        reflection=[(1000.0, "S")]
    )
    
    assert np.isclose(res.travel_times[0], 1.5, rtol=1e-5)

def test_pp_reflection_offset():
    """Test offset P-P reflection.
    Source: (0,0,0), Receiver: (2000,0,0)
    Reflector at 1000m.
    
    Note: The path is a symmetric triangle only because the entire down and up legs
    stay within a single constant-velocity layer (Layer 0, Vp=2000). 
    In a multi-layered medium above the reflector, the reflection point would still 
    be at mid-offset due to symmetry, but the legs would be refracted (bent).
    
    Geometry:
    Half-offset X/2 = 1000m
    Depth H = 1000m
    Angle theta = arctan(X/2 / H) = 45 degrees
    Leg length = sqrt(1000^2 + 1000^2) = 1414.21 m
    Total length = 2828.42 m
    Vp = 2000
    Time = 1.41421 s
    Ray parameter p = sin(theta) / Vp = sin(45) / 2000 = 0.7071 / 2000 = 0.00035355 s/m
    """
    X = 2000.0
    H = 1000.0
    Vp = 2000.0
    
    res = trace_rays(
        sources=[0,0,0],
        receivers=[X,0,0],
        velocity_df=test_model,
        source_phase="P",
        reflection=[(H, "P")]
    )
    
    expect_t = 2 * np.sqrt((X/2)**2 + H**2) / Vp
    theta = np.arctan2(X/2, H)
    expect_p = np.sin(theta) / Vp
   
    assert res.travel_times is not None
    assert np.isclose(res.travel_times[0], expect_t, rtol=1e-5)
    
    # Check ray parameter
    assert res.ray_parameters[0] > 0, "Ray parameter should be positive for offset ray"
    assert np.isclose(res.ray_parameters[0], expect_p, rtol=1e-5)

def test_multi_bounce():
    """Test multi-bounce: 0 -> 1000 -> 1 -> 1000 -> 0
    Total distance = 1000 + 999 + 999 + 1000 = 3998m
    Velocity = 2000
    Time = 1.999s
    """
    df = pd.DataFrame({
        "Depth": [0.0, 1.0, 1000.0],
        "Vp": [2000.0, 2000.0, 4000.0],
        "Vs": [1000.0, 1000.0, 2000.0],
        "Rho": [2000.0, 2000.0, 2500.0],
        "Qp": [500.0, 500.0, 1000.0],
        "Qs": [250.0, 250.0, 500.0],
    })
    res = trace_rays(
        sources=[0,0,0],
        receivers=[0,0,0],
        velocity_df=df,
        source_phase="P",
        reflection=[(1000.0, "P"), (1.0, "P"), (1000.0, "P")]
    )
    assert np.isclose(res.travel_times[0], 1.999, rtol=1e-5)

def test_invalid_phase():
    with pytest.raises(ValueError, match="Invalid phase"):
        trace_rays(
            sources=[0,0,0],
            receivers=[10,0,0],
            velocity_df=test_model,
            reflection=[(1000.0, "X")]
        )

def test_invalid_depth():
    with pytest.raises(ValueError, match="Invalid reflection depth"):
        trace_rays(
            sources=[0,0,0],
            receivers=[10,0,0],
            velocity_df=test_model,
            reflection=[(999.0, "P")]
        )

def test_surface_reflection_disallowed():
    """Verify that reflection at z=0.0 is explicitly disallowed."""
    with pytest.raises(ValueError, match="Reflection at the surface"):
        trace_rays(
            sources=[0,0,0],
            receivers=[0,0,0],
            velocity_df=test_model,
            source_phase="P",
            reflection=[(0.0, "P")]
        )

def test_reflection_at_source_depth_disallowed():
    """Verify that reflection at the source depth is disallowed."""
    with pytest.raises(ValueError, match="Cannot reflect at the starting depth"):
        trace_rays(
            sources=[0,0,1000],
            receivers=[0,0,1000],
            velocity_df=test_model,
            source_phase="P",
            reflection=[(1000.0, "P")]
        )

def test_refraction_vertical():
    """Test P-to-S refraction at 1000m, receiver at 2000m (in 2nd layer).
    Layer 2 extends to infinity, need 3rd row in model to bound it?
    build_layer_stack handles unbounded last layer.    
    """
    # Add a dummy 3rd layer so we can put receiver at 2000 solidly in layer 1
    # or just trust extrapolation.
    
    res = trace_rays(
        sources=[0,0,0],
        receivers=[0,0,2000], # Receiver at depth 2000
        velocity_df=test_model,
        source_phase="P",
        refraction=[(1000.0, "S")]
    )
    
    # Leg 1: 1000m / 2000 = 0.5
    # Leg 2: 1000m / 2000 = 0.5 (Vs of layer 1 is 2000)
    assert np.isclose(res.travel_times[0], 1.0, rtol=1e-5)

def test_transmission_amplitude():
    """Test amplitude product for vertical transmission through 3 layers.
    
    For a vertical ray (p=0), the Zoeppritz Tpp coefficient reduces exactly
    to the normal-incidence impedance formula, so we use transmission_normal
    to compute the expected value.
    """
    # Model:
    # 0-500: V=2000, Rho=2000 (Z=4e6)
    # 500-1000: V=3000, Rho=2500 (Z=7.5e6)
    # 1000-1500: V=2000, Rho=2000 (Z=4e6)
    
    df = pd.DataFrame({
        "Depth": [0.0, 500.0, 1000.0],
        "Vp": [2000.0, 3000.0, 2000.0],
        "Vs": [1000.0, 1500.0, 1000.0],
        "Rho": [2000.0, 2500.0, 2000.0],
        "Qp": [400.0, 600.0, 400.0],
        "Qs": [200.0, 300.0, 200.0],
    })
    
    # Trace vertical ray (p=0) through all interfaces
    # Source 0, Receiver 1200
    res = trace_rays(
         sources=[0,0,0],
         receivers=[0,0,1200],
         velocity_df=df,
         source_phase="P",
         compute_amplitude=True,
         transcoef_method="standard"
    )
    
    # At p=0 the Zoeppritz Tpp equals the impedance formula
    t1 = transmission_normal(2000, 2000, 3000, 2500)
    t2 = transmission_normal(3000, 2500, 2000, 2000)
    expected = abs(t1 * t2)
    
    assert np.isclose(res.trans_product[0], expected, rtol=1e-4)


def test_transmission_normalized():
    """Normalized transmission product should differ from standard for offset rays."""
    df = pd.DataFrame({
        "Depth": [0.0, 1000.0, 2000.0],
        "Vp": [3000.0, 5000.0, 4000.0],
        "Vs": [1500.0, 2500.0, 2000.0],
        "Rho": [2200.0, 2600.0, 2400.0],
        "Qp": [300.0, 500.0, 400.0],
        "Qs": [150.0, 250.0, 200.0],
    })

    # Offset ray so p != 0
    res_std = trace_rays(
        sources=[0, 0, 0],
        receivers=[3000, 0, 2500],
        velocity_df=df,
        source_phase="P",
        compute_amplitude=True,
        transcoef_method="standard",
    )
    res_norm = trace_rays(
        sources=[0, 0, 0],
        receivers=[3000, 0, 2500],
        velocity_df=df,
        source_phase="P",
        compute_amplitude=True,
        transcoef_method="normalized",
    )

    # Both should be finite and positive
    assert np.isfinite(res_std.trans_product[0])
    assert np.isfinite(res_norm.trans_product[0])
    assert res_std.trans_product[0] > 0
    assert res_norm.trans_product[0] > 0

    # They should differ for non-vertical rays
    assert not np.isclose(res_std.trans_product[0], res_norm.trans_product[0], rtol=1e-6)

    # Travel times and ray parameters must be identical (kinematics unchanged)
    assert np.isclose(res_std.travel_times[0], res_norm.travel_times[0], rtol=1e-10)
    assert np.isclose(res_std.ray_parameters[0], res_norm.ray_parameters[0], rtol=1e-10)


def test_normalized_vertical_ray():
    """At normal incidence (p=0), normalized T equals standard T * sqrt(Z_out/Z_in).
    
    The Červený (2001) normalization factor at p=0 simplifies to
    sqrt(v_out * rho_out / (v_in * rho_in)) = sqrt(Z_out / Z_in).
    """
    from laytracer.amplitude import psv_rt_coefficients, normalize_rt_coefficient

    df = pd.DataFrame({
        "Depth": [0.0, 1000.0],
        "Vp": [3000.0, 5000.0],
        "Vs": [1500.0, 2500.0],
        "Rho": [2200.0, 2600.0],
        "Qp": [300.0, 500.0],
        "Qs": [150.0, 250.0],
    })

    res_std = trace_rays(
        sources=[0, 0, 0],
        receivers=[0, 0, 1500],
        velocity_df=df,
        source_phase="P",
        compute_amplitude=True,
        transcoef_method="standard",
    )
    res_norm = trace_rays(
        sources=[0, 0, 0],
        receivers=[0, 0, 1500],
        velocity_df=df,
        source_phase="P",
        compute_amplitude=True,
        transcoef_method="normalized",
    )

    # Verify normalized = standard * sqrt(Z_out / Z_in) at p=0
    Z_in = 3000.0 * 2200.0
    Z_out = 5000.0 * 2600.0
    expected_norm = res_std.trans_product[0] * np.sqrt(Z_out / Z_in)
    assert np.isclose(res_norm.trans_product[0], expected_norm, rtol=1e-6)

if __name__ == "__main__":
    # Manually run if executed as script
    try:
        test_pp_reflection_vertical()
        test_ps_reflection_vertical()
        test_pp_reflection_offset()
        test_multi_bounce()
        test_invalid_phase()
        test_invalid_depth()
        test_refraction_vertical()
        test_transmission_amplitude()
        print("All tests passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
