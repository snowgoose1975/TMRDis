"""Portable command line for the frozen paper-v1 prediction and evaluation protocol."""
from __future__ import annotations
import argparse
from importlib import import_module
from pathlib import Path
import shutil
import sys
import sysconfig

COMMANDS = {
    'prepare': ('paper_inference', 'prepare_main'),
    'prepare-training': ('paper_dataset', 'prepare_training_main'),
    'predict': ('paper_inference', 'predict_main'),
    'verify-models': ('paper_inference', 'verify_models_main'),
    'train': ('paper_training', 'train_main'),
    'train-matrix': ('paper_training', 'train_matrix_main'),
    'evaluate': ('paper_replay', 'evaluate_main'),
    'replay': ('paper_replay', 'replay_main'),
    'export-resources': ('paper_replay', 'export_main'),
}

def config_directory():
    """Locate either a source checkout's config or installed wheel data files."""
    parent = Path(__file__).resolve().parent.parent
    candidates = [parent / 'configs',
                  Path(sysconfig.get_path('data')) / 'share/zchen-disorder-predictor/configs',
                  parent / 'share/zchen-disorder-predictor/configs']
    for candidate in candidates:
        if (candidate / 'paper-v1.json').is_file():
            return candidate
    raise FileNotFoundError('Paper configuration files missing; install the complete wheel or supply config files from the source release.')

def config_main(argv=None):
    parser = argparse.ArgumentParser(description='Export installed paper-v1 configurations to a new directory')
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args(argv)
    source = config_directory()
    destination = Path(args.output_dir)
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite existing directory: {destination}')
    shutil.copytree(source, destination)
    print(destination.resolve())
    return 0

def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__, prog='disorder-paper')
    parser.add_argument('command', nargs='?', choices=[*COMMANDS, 'config'])
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    if not argv or argv[0] in ('-h', '--help'):
        parser.print_help()
        print('\nRun disorder-paper COMMAND --help for individual arguments.')
        return 0
    if argv[0] == '--version':
        from importlib.metadata import version
        print(version('my-disorder-predictor'))
        return 0
    command, rest = argv[0], argv[1:]
    if command == 'config':
        return config_main(rest)
    if command not in COMMANDS:
        parser.error(f'unknown command: {command}')
    module_name, function_name = COMMANDS[command]
    function = getattr(import_module('my_predictor.' + module_name), function_name)
    result = function(rest)
    return result if isinstance(result, int) else 0

if __name__ == '__main__':
    main()
