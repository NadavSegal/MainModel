import numpy as np

def get_matrix_accuracy(procces_frame_output, src_labels):
    procces_frame_output[src_labels <= 0] = -1
        
    true_mask = procces_frame_output == src_labels
        
    true_mask = np.sum(true_mask) / np.sum(src_labels>0)
        
    return true_mask, 1 - true_mask

def img_and_labels_to_model_input_format(x=None, y=None):
        
        if not x is None:
            x = np.transpose(x,(2,0,1))
            x = x.astype('float32') 
            x = x / 255.0
            x = x[None,...]
            
        if not y is None:
            y = y[..., None]
            y = np.transpose(y,(2, 0, 1))
            y = y[None,...]
#            y = np.swapaxes(y, -1, 1)
                
        return x, y