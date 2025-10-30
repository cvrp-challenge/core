#!/usr/bin/env python3
"""
Unified Solver Runner for CVRP Challenge
Run any configured solver with a specified instance.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any
import yaml


class SolverRunner:
    """Manages and executes VRP solvers."""
    
    def __init__(self, config_path: str = "config/solvers.yaml"):
        self.root_dir = Path(__file__).parent.absolute()
        self.config_path = self.root_dir / config_path
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load solver configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if not config or 'solvers' not in config:
            raise ValueError("Invalid configuration file: missing 'solvers' section")
        
        return config['solvers']
    
    def list_solvers(self) -> None:
        """Print all available solvers with their status."""
        print("\n" + "="*70)
        print("Available Solvers".center(70))
        print("="*70 + "\n")
        
        enabled_solvers = []
        disabled_solvers = []
        
        for solver_id, solver_info in self.config.items():
            status = "✓ ENABLED" if solver_info.get('enabled', False) else "✗ DISABLED"
            solver_entry = {
                'id': solver_id,
                'name': solver_info.get('name', solver_id),
                'description': solver_info.get('description', ''),
                'type': solver_info.get('type', 'unknown'),
                'status': status,
                'notes': solver_info.get('notes', '')
            }
            
            if solver_info.get('enabled', False):
                enabled_solvers.append(solver_entry)
            else:
                disabled_solvers.append(solver_entry)
        
        # Print enabled solvers
        for solver in enabled_solvers:
            print(f"[{solver['status']}] {solver['id']}")
            print(f"  Name: {solver['name']}")
            print(f"  Type: {solver['type']}")
            print(f"  Description: {solver['description']}")
            if solver['notes']:
                print(f"  Notes: {solver['notes']}")
            print()
        
        # Print disabled solvers
        if disabled_solvers:
            print("-" * 70)
            print("Disabled Solvers:".center(70))
            print("-" * 70 + "\n")
            for solver in disabled_solvers:
                print(f"[{solver['status']}] {solver['id']}")
                print(f"  Name: {solver['name']}")
                if solver['notes']:
                    print(f"  Notes: {solver['notes']}")
                print()
        
        print("="*70)
        print(f"\nTo run a solver: python runner.py <solver_id> <instance_path>")
        print(f"Example: python runner.py hgs instances/test-instances/x/X-n101-k25.vrp\n")
    
    def validate_solver(self, solver_id: str) -> Dict[str, Any]:
        """Validate and return solver configuration."""
        if solver_id not in self.config:
            available = ', '.join(self.config.keys())
            raise ValueError(
                f"Unknown solver: '{solver_id}'\n"
                f"Available solvers: {available}\n"
                f"Run with --list to see all solvers."
            )
        
        solver_info = self.config[solver_id]
        
        if not solver_info.get('enabled', False):
            raise ValueError(
                f"Solver '{solver_id}' is disabled.\n"
                f"Reason: {solver_info.get('notes', 'No reason provided')}"
            )
        
        return solver_info
    
    def validate_instance(self, instance_path: str, solver_info: Dict[str, Any]) -> Path:
        """Validate instance file exists and has correct format."""
        instance = Path(instance_path)
        
        # Try relative to root directory if not found
        if not instance.exists():
            instance = self.root_dir / instance_path
        
        if not instance.exists():
            raise FileNotFoundError(f"Instance file not found: {instance_path}")
        
        # Check file extension
        supported_formats = solver_info.get('supported_formats', [])
        if supported_formats and instance.suffix not in supported_formats:
            raise ValueError(
                f"Unsupported file format: {instance.suffix}\n"
                f"Solver supports: {', '.join(supported_formats)}"
            )
        
        return instance
    
    def build_command(self, solver_info: Dict[str, Any], instance: Path, 
                     extra_args: Dict[str, Any]) -> list:
        """Build the command to execute the solver."""
        executable = solver_info['executable']
        
        # Make executable path absolute relative to root
        if not Path(executable).is_absolute():
            executable = str(self.root_dir / executable)
        
        # Check if executable exists (for compiled solvers)
        if solver_info['type'] == 'compiled' and not Path(executable).exists():
            raise FileNotFoundError(
                f"Solver executable not found: {executable}\n"
                f"Have you built the solver? See COMMANDS.md for build instructions."
            )
        
        # Get default arguments and merge with extra args
        default_args = solver_info.get('default_args', {})
        if isinstance(default_args, list):
            # If default_args is a list, convert to dict
            args = {}
        else:
            args = default_args.copy() if default_args else {}
        
        # Merge with extra args
        args.update(extra_args)
        
        # Build command from template
        template = solver_info.get('command_template', '{executable} {instance}')
        
        # Prepare template variables
        template_vars = {
            'executable': executable,
            'instance': str(instance),
            **args
        }
        
        # Simple template substitution
        command_str = template.format(**template_vars)
        
        # Split into command list
        command = command_str.split()
        
        return command
    
    def run(self, solver_id: str, instance_path: str, 
            extra_args: Dict[str, Any] = None, 
            verbose: bool = True) -> int:
        """
        Run a solver on an instance.
        
        Args:
            solver_id: ID of the solver to run
            instance_path: Path to the instance file
            extra_args: Additional arguments to pass to the solver
            verbose: Whether to print execution details
            
        Returns:
            Exit code from the solver
        """
        if extra_args is None:
            extra_args = {}
        
        # Validate inputs
        solver_info = self.validate_solver(solver_id)
        instance = self.validate_instance(instance_path, solver_info)
        
        # Build command
        command = self.build_command(solver_info, instance, extra_args)
        
        if verbose:
            print("\n" + "="*70)
            print(f"Running {solver_info['name']}".center(70))
            print("="*70)
            print(f"\nSolver:   {solver_id}")
            print(f"Instance: {instance.name}")
            print(f"Path:     {instance}")
            print(f"Command:  {' '.join(command)}")
            print("\n" + "-"*70 + "\n")
        
        # Execute solver
        try:
            result = subprocess.run(
                command,
                cwd=str(self.root_dir),
                check=False
            )
            
            if verbose:
                print("\n" + "-"*70)
                print(f"Solver finished with exit code: {result.returncode}")
                print("="*70 + "\n")
            
            return result.returncode
            
        except KeyboardInterrupt:
            print("\n\nExecution interrupted by user.")
            return 130
        except Exception as e:
            print(f"\nError executing solver: {e}", file=sys.stderr)
            return 1


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Unified solver runner for CVRP Challenge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available solvers
  python runner.py --list
  
  # Run HGS on an instance
  python runner.py hgs instances/test-instances/x/X-n101-k25.vrp
  
  # Run FILO2 on an instance
  python runner.py filo2 instances/test-instances/x/X-n101-k25.vrp
  
  # Run PyVRP with custom seed and runtime
  python runner.py pyvrp instances/test-instances/x/X-n101-k25.vrp --seed 42 --max_runtime 60
        """
    )
    
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List all available solvers and exit'
    )
    
    parser.add_argument(
        'solver',
        nargs='?',
        help='Solver ID to use (e.g., hgs, filo2, pyvrp)'
    )
    
    parser.add_argument(
        'instance',
        nargs='?',
        help='Path to instance file'
    )
    
    parser.add_argument(
        '--config',
        default='config/solvers.yaml',
        help='Path to solver configuration file (default: config/solvers.yaml)'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress runner output (solver output still shown)'
    )
    
    # Allow arbitrary extra arguments
    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed for the solver'
    )
    
    parser.add_argument(
        '--max_runtime',
        type=int,
        help='Maximum runtime in seconds'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    try:
        runner = SolverRunner(config_path=args.config)
        
        # List solvers if requested
        if args.list:
            runner.list_solvers()
            return 0
        
        # Validate required arguments
        if not args.solver or not args.instance:
            print("Error: Both solver and instance are required.", file=sys.stderr)
            print("Run with --help for usage information.", file=sys.stderr)
            return 1
        
        # Collect extra arguments
        extra_args = {}
        if args.seed is not None:
            extra_args['seed'] = args.seed
        if args.max_runtime is not None:
            extra_args['max_runtime'] = args.max_runtime
        
        # Run solver
        return runner.run(
            args.solver,
            args.instance,
            extra_args=extra_args,
            verbose=not args.quiet
        )
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())

