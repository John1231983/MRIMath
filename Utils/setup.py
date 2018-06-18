import pip
from HardwareHandler import HardwareHandler
def import_or_install(package):
    try:
        __import__(package)
    except ImportError:
        pip.main(['install', package])  


hardwareHandler = HardwareHandler()

import_or_install("nibabel")
import_or_install("requests")
import_or_install("multiprocessing")
if hardwareHandler.getAvailableGPUs() > 0:
  import_or_install("tensorflow-gpu")
else:
  import_or_install("tensorflow")
import_or_install("keras")
import_or_install("opencv-python")
import_or_install("matplotlib")
import_or_install("numpy")
import_or_install("scipy")
