"""
New Entry Point for TEKNOFEST Simulation.
Launches the simulation directly with 'straight_attack' scenario.
"""
import sys
import os
from pathlib import Path

# Add project root to python path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

# Import the main function from scripts.run
from scripts.run import main

if __name__ == "__main__":
    print(f"🚀 Launching Straight Attack Scenario...")
    print(f"📂 Project Root: {project_root}")
    
    # Override command line arguments to force specific configuration
    # We manipulate sys.argv so that argparse in scripts.run sees these arguments
    sys.argv = [
        sys.argv[0],          # Script name
        "--mode", "ui",       # UI Mode
        "--scenario", "straight_attack", # Requested Scenario
        "--gl-view",          # Enable OpenGL 3D View Overlay
        "--duration", "300"   # 5 Minutes Max Duration
    ]
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Simulation wrapper stopped.")
    except Exception as e:
        print(f"\n❌ Error in simulation wrapper: {e}")
        import traceback
        traceback.print_exc()
