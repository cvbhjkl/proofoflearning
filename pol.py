from hashlib import sha256
import time
def check_require(parameter_prev,parameter_now,requirements):
    i=0
    for require in requirements:
        if require%2==0:
            if parameter_now[i]>=parameter_prev[i]:
                return False
        else:
            if parameter_now[i]<=parameter_prev[i]:
                return False   
        i+=1
    return True         

def save_parameters(parameters_enum,requirements):
    saved=[]
    for require in requirements:   ##TODO：sort requirements to make read_parameter faster
        saved.append(read_parameter(parameters_enum(),require//2))
    return saved

def read_parameter(parameters,index):
    count=0
    for parameter_tensor in parameters:
      size_tensor=get_tensor_size(parameter_tensor.size())
      if count+size_tensor>index:
        x=parameter_tensor.flatten()
        return x[index-count].item()
      count+=size_tensor

def get_tensor_size(tensor_size):
  s=1
  for i in tensor_size:
    s*=i
  return s

def get_net_size(parameters):
  count=0
  for parameter_tensor in parameters:
    count+=get_tensor_size(parameter_tensor.size())
  return count

##for requirements, even number means decrease and odd means increase, number/2 specify the parameter
def get_requirements(net_size,difficulty,hash_prev):
  list_requirements=[]
  h1=hash_prev
  for i in range(difficulty):
    while True:
      h1=sha256(h1.encode("utf-8")).hexdigest()
      next_require=int(h1,16)%(2*net_size)
      ##collision avoidance
      if next_require%2==1: 
        if next_require in list_requirements or next_require-1 in list_requirements:
          continue
      else:
        if next_require in list_requirements or next_require+1 in list_requirements:
          continue       
      break
    list_requirements.append(int(h1,16)%(2*net_size))  
  return list_requirements

##get hash_prev from the previous block
def get_parameter_hash(parameter_prev):
    h0=""
    for i in parameter_prev:
        h0=sha256((str(i)+h0).encode("utf-8")).hexdigest()
    return h0

def show_requirements(requirements):
    print("Requirements now:")
    for j in requirements:
        if j%2==1:
          print('%d should increase'%(j/2))
        else:
          print('%d should decrease'%(j/2))

def stop_running(pct,running_time):
    time_now=time.time()
    stop_time=(time_now-running_time)*(100.0-pct)/pct
    print("already run %f seconds"%(time_now-running_time))
    print("now sleep for %f seconds"%stop_time)
    time.sleep(stop_time)