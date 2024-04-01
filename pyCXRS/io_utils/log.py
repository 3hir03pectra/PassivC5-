import numpy as np

def read_log(file_name):
    # Read text file with strict form:
    #       good = 1
    #       frame_start = 101
    #       frame_end = 177
    #       active_frames = [104, 114, 124, 134, 144, 154, 164, 174]
    #       central_pixel = 231.041521
    header={}
    with open(file_name,'rb') as f:
        lines=f.readlines()
    header['good']=bool(lines[0].decode().split('=')[1].strip())
    header['frame_start']=int(lines[1].decode().split('=')[1].strip())
    header['frame_end']=int(lines[2].decode().split('=')[1].strip())
    af=lines[3].decode().split('=')[1].strip()
    af=af.replace('[','')
    af=af.replace(']','')
    header['active_frames']=np.fromstring(af,dtype=int, sep=',')
    header['central_pixel']=float(lines[4].decode().split('=')[1].strip())
    return header

