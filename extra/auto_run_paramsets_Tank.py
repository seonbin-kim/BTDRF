import os
import threading, queue
import numpy as np
import time


def getFolderLocker(logFolder):
    while True:
        try:
            os.makedirs(logFolder+"/lockFolder")
            break
        except: 
            time.sleep(0.01)

def releaseFolderLocker(logFolder):
    os.removedirs(logFolder+"/lockFolder")

def getStopFolder(logFolder):
    return os.path.isdir(logFolder+"/stopFolder")


def get_param_str(key, val):
    if key == 'data_name':
        return f'--datadir {datafolder}/{val} '
    else:
        return f'--{key} {val} '

def get_param_list(param_dict):
    param_keys = list(param_dict.keys())
    param_modes = len(param_keys)
    param_nums = [len(param_dict[key]) for key in param_keys]
    
    param_ids = np.zeros(param_nums+[param_modes], dtype=int)
    for i in range(param_modes):
        broad_tuple = np.ones(param_modes, dtype=int).tolist()
        broad_tuple[i] = param_nums[i]
        broad_tuple = tuple(broad_tuple)
        print(broad_tuple)
        param_ids[...,i] = np.arange(param_nums[i]).reshape(broad_tuple)
    param_ids = param_ids.reshape(-1, param_modes)
    # print(param_ids)
    print(len(param_ids))
    
    params = []
    expnames = []
    for i in range(param_ids.shape[0]):
        one = ""
        name = ""
        param_id = param_ids[i]
        for j in range(param_modes):
            key = param_keys[j]
            val = param_dict[key][param_id[j]]
            if type(key) is tuple:
                assert len(key) == len(val)
                for k in range(len(key)):
                    one += get_param_str(key[k], val[k])
                    name += f'{val[k]},'
                name=name[:-1]+'-'
            else:
                one += get_param_str(key, val)
                name += f'{val}-'
        params.append(one)
        name=name.replace(' ','')
        print(name)
        expnames.append(name[:-1])
    # print(params)
    return params, expnames







if __name__ == '__main__':


    # tankstemple
    expFolder = "tankstemple_4/"
    datafolder = '/home/ksb/TensoRF_Transformer_Triplane_Explicit_test/data/TanksAndTemple/'
    param_dict = {
                # 'data_name': ['Truck','Barn','Caterpillar','Family','Ignatius'],
                'data_name': ['Truck', 'Family', 'Ignatius'],
                'core_num': [7],
                'num_level': [8]
        }



    #setting available gpus
    gpus_que = queue.Queue(3)
    for i in [3]:
        gpus_que.put(i)
    
    os.makedirs(f"log/{expFolder}", exist_ok=True)

    def run_program(gpu, expname, param):
        cmd = f'OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES={gpu}  /home/ksb/miniconda3/envs/TensoRF/bin/python3 ../train_4.py ' \
            f'--expname {expname} --basedir ./log/{expFolder} --config ../configs/truck.txt ' \
            f'{param}' \
            f'> "log/{expFolder}{expname}/{expname}.txt"'
        print(cmd)
        os.system(cmd)
        gpus_que.put(gpu)

    params, expnames = get_param_list(param_dict)

    
    logFolder=f"log/{expFolder}"
    os.makedirs(logFolder, exist_ok=True)

    ths = []
    for i in range(len(params)):

        if getStopFolder(logFolder):
            break


        targetFolder = f"log/{expFolder}{expnames[i]}"
        gpu = gpus_que.get()
        getFolderLocker(logFolder)
        if os.path.isdir(targetFolder):
            releaseFolderLocker(logFolder)
            gpus_que.put(gpu)
            continue
        else:
            os.makedirs(targetFolder, exist_ok=True)
            print("making",targetFolder, "running",expnames[i], params[i])
            releaseFolderLocker(logFolder)


        t = threading.Thread(target=run_program, args=(gpu, expnames[i], params[i]), daemon=True)
        t.start()
        ths.append(t)
    
    for th in ths:
        th.join()