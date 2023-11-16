import yaml, os
import numpy as np
from multiprocessing import Process, Manager
from calibrate import calibrate
from detect import record, detect
    
def main():
    # 1. Load calibration data
    # If not exist, then create one
    if os.path.exists('calibration.yaml'):
        with open('calibration.yaml', 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
        mtx, dist = np.array(data['camera_matrix']), np.array(data['dist_coeff'])
    else:
        mtx, dist = calibrate()

    # 2. Record
    manager = Manager()
    images = manager.list()

    p1 = Process(target=record, args=[mtx, dist, images])
    p2 = Process(target=detect, args=[images])

    p1.start()
    p2.start()

    p1.join()
    p2.join()
    
if __name__ == '__main__':
    main()
