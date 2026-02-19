
import pytest
import numpy as np
import pandas as pd
from laytracer import trace_rays

# Simple 2-layer model
# Layer 0: Vp=2000, Vs=1000, Rho=2000, Depth=0
# Layer 1: Vp=4000, Vs=2000, Rho=2500, Depth=1000
test_model = pd.DataFrame({
    "Depth": [0.0, 1000.0],
    "Vp": [2000.0, 4000.0],
    "Vs": [1000.0, 2000.0],
    "Rho": [2000.0, 2500.0]
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
    Path is symmetric triangle.
    Half-path: horizontal=1000, vertical=1000.
    Leg length = sqrt(1000^2 + 1000^2) = 1414.21 m
    Total length = 2828.42 m
    Vp = 2000
    Time = 1.41421 s
    """
    res = trace_rays(
        sources=[0,0,0],
        receivers=[2000,0,0],
        velocity_df=test_model,
        source_phase="P",
        reflection=[(1000.0, "P")]
    )
    expect = 2 * np.sqrt(1000**2 + 1000**2) / 2000.0
    print(f"DEBUG: Calculated T={res.travel_times[0]}, Expected={expect}")
    print(f"DEBUG: Ray Param p={res.ray_parameters[0]}")
    assert np.isclose(res.travel_times[0], expect, rtol=1e-4)

def test_multi_bounce():
    """Test P-P-P-P bounce: 0 -> 1000 -> 0 -> 1000 -> 0
    Total distance = 4000m
    Velocity = 2000
    Time = 2.0s
    Vertical ray
    """
    res = trace_rays(
        sources=[0,0,0],
        receivers=[0,0,0],
        velocity_df=test_model,
        source_phase="P",
        reflection=[(1000.0, "P"), (0.0, "P"), (1000.0, "P")]
    )
    # Note: Reflector at 0.0 is surface multiple
    assert np.isclose(res.travel_times[0], 2.0, rtol=1e-5)

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

def test_refraction_vertical():
    """Test P-to-S refraction at 1000m, receiver at 2000m (in 2nd layer).
    Layer 2 extends to infinity, need 3rd row in model to bound it?
    build_layer_stack handles unbounded last layer.
    
    Path: 
    0->1000 (Layer 0, P, V=2000): t = 0.5s
    1000->2000 (Layer 1, S, V=2000): t = 0.5s
    Total = 1.0s
    Wait, Layer 1 Vs is 2000.
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
    """Test amplitude product for normal transmission through 3 layers."""
    # Model:
    # 0-500: V=2000, Rho=2000 (Z=4e6)
    # 500-1000: V=3000, Rho=2500 (Z=7.5e6)
    # 1000-1500: V=2000, Rho=2000 (Z=4e6)
    
    df = pd.DataFrame({
        "Depth": [0.0, 500.0, 1000.0],
        "Vp": [2000.0, 3000.0, 2000.0],
        "Vs": [1000.0, 1500.0, 1000.0],
        "Rho": [2000.0, 2500.0, 2000.0]
    })
    
    # Trace vertical ray (p=0) through all interfaces
    # Source 0, Receiver 1200
    res = trace_rays(
         sources=[0,0,0],
         receivers=[0,0,1200],
         velocity_df=df,
         source_phase="P",
         compute_amplitude=True,
         transcoef_method="normal"
    )
    
    # Expected:
    # Int 1 (500m): T1 = 2*Z2/(Z1+Z2) = 2*7.5/(4+7.5) = 15/11.5
    # Int 2 (1000m): T2 = 2*Z3/(Z2+Z3) = 2*4/(7.5+4) = 8/11.5
    
    t1 = 15.0/11.5
    t2 = 8.0/11.5
    expected = t1 * t2
    
    print(f"DEBUG: TransProd Calculated={res.trans_product[0]}")
    print(f"DEBUG: TransProd Expected={expected}")
    
    assert np.isclose(res.trans_product[0], expected, rtol=1e-4)

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
