import warnings
# check for installation of torch-scatter
try:
    import torch_scatter
except:
    warnings.warn("torch-scatter is either not installed or not properly installed; please check for the appropriate version", UserWarning)
    raise Exception(
        "see pyproject.toml for torch-scatter + PyG find-links (torch/cu121 must match)"
    )
    # cmd = ["pip", "install", "torch-scatter", "-f", "https://data.pyg.org/whl/torch-1.13.1+cu117.html"]
    # print("try downloading torch-scatter")
    # subprocess.call(cmd)

# set this to true to start the debugger on any exception
DEBUG_MODE = False
if DEBUG_MODE:
    import pdb
    import sys
    import traceback

    def debughook(etype, value, tb):
        traceback.print_exception(etype, value, tb)
        print()
        pdb.pm()

    sys.excepthook = debughook
