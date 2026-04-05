import sys
import importlib

try:
    importlib.import_module("server.app")
    print("Success!")
except BaseException as e:
    import traceback
    traceback.print_exc()
