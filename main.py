#!/usr/bin/env python3
"""
Teknofest Savaşan İHA Simülasyonu (vNext Wrapper)

Wrapper script for backward compatibility.
Uses scripts.run.SimulationRunner internally.
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.run import SimulationRunner, load_scenario

def main():
    parser = argparse.ArgumentParser(description='Teknofest Savaşan İHA Simülasyonu')
    parser.add_argument('--config', '-c', type=str, help='Konfigürasyon dosyası')
    parser.add_argument('--scenario', '-s', type=str, default='default',
                       help='Senaryo seçimi')
    parser.add_argument('--uav-count', '-n', type=int, default=3,
                       help='Düşman İHA sayısı (Overrides scenario config)')
    parser.add_argument('--model', '-m', type=str,
                       help='Harici YOLO model dosyası')
    parser.add_argument('--run-id', type=str, default=None,
                        help='Run ID')
    parser.add_argument('--duration', '-d', type=float,
                        help='Simülasyon süresi (saniye)')
    
    args = parser.parse_args()
    
    # Check if we should show the menu (no args provided or explicit --menu?)
    # Ideally if user runs `python main.py` without args, we show menu.
    # argparse sets defaults, so we need to check if parameters were defaulted.
    # Simplest check: len(sys.argv) == 1
    
    use_menu = (len(sys.argv) == 1)
    
    config = {}
    
    if use_menu:
        try:
            from src.simulation.ui.launcher import Launcher
            launcher = Launcher()
            launcher_config = launcher.loop()
            
            if not launcher_config:
                print("❌ Operasyon iptal edildi.")
                sys.exit(0)
            
            # Helper to load existing scenario config
            try:
                scenario_config = load_scenario(launcher_config['scenario'])
            except (FileNotFoundError, ValueError) as exc:
                print(f"❌ {exc}")
                sys.exit(1)
            
            config = {
                **scenario_config,
                'scenario': launcher_config['scenario'],
                'seed': launcher_config['seed'],
                'duration': launcher_config['duration'],
                'mode': launcher_config['mode'],
                'run_id': None
            }
            # Update running mode based on launcher selection
            # We need to pass this mode to SimulationRunner
            
        except ImportError as e:
            print(f"Launcher module not found: {e}")
            return
        except Exception as e:
            print(f"Launcher error: {e}")
            sys.exit(1)
    else:
        # Load scenario config
        try:
            scenario_config = load_scenario(args.scenario)
        except (FileNotFoundError, ValueError) as exc:
            print(f"❌ {exc}")
            sys.exit(1)
        
        # Construct base config
        config = {
            **scenario_config,
            'scenario': args.scenario,
            'seed': 42, # Default seed for main.py users
            'run_id': args.run_id
        }
        
        # Overrides from CLI
        if args.uav_count is not None:
            config['enemy_count'] = args.uav_count
            
        if args.model:
            print("⚠️  UYARI: vNext simülasyonu deterministik 'SimulationDetector' kullanır.")
            print("    '--model' argümanı yoksayılacak (sentetik tespitler kullanılacak).")
            
        if args.duration:
            config['duration'] = args.duration
        
    # Run with configured mode (default UI)
    run_mode = config.get('mode', 'ui')
    runner = SimulationRunner(config, mode=run_mode)
    
    try:
        runner.run()
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
