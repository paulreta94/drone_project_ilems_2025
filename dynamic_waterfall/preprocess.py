import numpy as np
class Smatrix:
    def __init__(self,data):
        self.data=data
    def slice_matrix(self,frame_width):
        total_width = np.shape(self.T)[1]
        new_matrix = self.T[:,:frame_width]
        for i in range(total_width//frame_width-2):
            np.tile(new_matrix,s.T[:,i*frame_width:(i+1)*frame_width])
            new_s=np.array([s.T[:,:2],s.T[:,2:4],s.T[:,4:6],s.T[:,6:8],s.T[:,8:]])
            
