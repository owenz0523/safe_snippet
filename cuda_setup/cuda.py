import torch

##### Check cuda.txt if it is not running in GPU
def cudaCheck():
    try:
        print("Is torch cuda available?", torch.cuda.is_available())
        if torch.cuda.is_available():
            dev = "cuda"
        else:
            dev = "cpu"
    except:
        dev = "cpu"
        print('cudaCheck error')
    print("device running:", dev)
    return dev