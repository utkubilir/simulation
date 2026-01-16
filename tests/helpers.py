
import subprocess
import shutil
import hashlib
import json
import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

# Ensure scripts module is accessible
sys.path.insert(0, str(Path(__file__).parent.parent))

@dataclass
class RunMetrics:
    duration: float
    total_frames: int
    correct_locks: int
    incorrect_locks: int
    final_score: float
    output_dir: Path
    frames_path: Path
    metrics_path: Path
    
    @property
    def has_output(self) -> bool:
        return self.frames_path.exists() and self.metrics_path.exists()

def get_project_root() -> Path:
    return Path(__file__).parent.parent

def sha256_file(path: Path) -> str:
    """Calculate SHA256 of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()

def find_latest_run_dir(base_output: Path) -> Optional[Path]:
    """Find the most recent run subdirectory in output dir."""
    if not base_output.exists():
        return None
    subdirs = [d for d in base_output.iterdir() if d.is_dir()]
    if not subdirs:
        return None
    return sorted(subdirs, key=lambda d: d.stat().st_mtime)[-1]

def load_metrics_json(path: Path) -> Dict[str, Any]:
    """Load metrics.json as dict."""
    with open(path) as f:
        return json.load(f)

def run_sim_headless(
    scenario: str, 
    seed: int, 
    duration: float = 5.0,
    output_temp_dir: str = "/tmp/sim_test",
    run_id: Optional[str] = None
) -> RunMetrics:
    """
    Run simulation in a subprocess (headless) and return metrics paths.
    Using subprocess ensures clean state and avoids global variable leakage.
    """
    root = get_project_root()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root)
    env["SDL_VIDEODRIVER"] = "dummy"
    
    cmd = [
        sys.executable,
        "-m", "scripts.run",
        "--mode", "headless",
        "--scenario", scenario,
        "--seed", str(seed),
        "--duration", str(duration),
        "--output", output_temp_dir
    ]
    
    if run_id:
        cmd.extend(["--run-id", run_id])
        
    result = subprocess.run(
        cmd, 
        cwd=root,
        capture_output=True,
        text=True,
        env=env
    )
    
    if result.returncode != 0:
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        raise RuntimeError(f"Simulation failed with exit code {result.returncode}")

    # Locate output
    if run_id:
        run_dir = Path(output_temp_dir) / run_id
    else:
        run_dir = find_latest_run_dir(Path(output_temp_dir))
        
    if not run_dir or not run_dir.exists():
         raise FileNotFoundError(f"Could not find run directory in {output_temp_dir}")
         
    metrics_json = run_dir / "metrics.json"
    frames_jsonl = run_dir / "frames.jsonl"
    
    # Load metrics for convenience
    m_data = {}
    if metrics_json.exists():
        m_data = load_metrics_json(metrics_json)
        
    return RunMetrics(
        duration=m_data.get('duration', 0.0),
        total_frames=m_data.get('total_frames', 0),
        correct_locks=m_data.get('correct_locks', 0),
        incorrect_locks=m_data.get('incorrect_locks', 0),
        final_score=m_data.get('final_score', 0.0),
        output_dir=run_dir,
        frames_path=frames_jsonl,
        metrics_path=metrics_json
    )

def validate_jsonl_schema(path: Path, required_keys: List[str]):
    """Validate that every line in JSONL has keys."""
    if not path.exists():
        raise FileNotFoundError(f"{path} missing")
        
    with open(path) as f:
        for idx, line in enumerate(f):
            if not line.strip(): continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON at line {idx+1}")
                
            for k in required_keys:
                if k not in data:
                    raise KeyError(f"Line {idx+1} missing key '{k}'")
