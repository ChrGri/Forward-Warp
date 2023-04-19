import torch
from torch.nn import Module, Parameter
from torch.autograd import Function

# custom autograd function
# example see https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
class Forward_Warp_Python:
    @staticmethod
    def forward(im0, flow, im1):
        
        # initialize target image
        #im1 = torch.zeros( (im0.shape[0], im0.shape[1], H_t, W_t) )     
        
        B_s = im0.shape[0]
        H_s = im0.shape[2]
        W_s = im0.shape[3]
        
        B_t = im1.shape[0]
        C_t = im1.shape[1]
        H_t = im1.shape[2]
        W_t = im1.shape[3]
        
        round_flow = torch.round(flow)
        
        # iterate over all pixels in source image
        for b in range(B_s):
            for h in range(H_s):
                for w in range(W_s):
                    
                    # get pixel position in target image
                    x = int(round_flow[b, h, w, 0])
                    y = int(round_flow[b, h, w, 1])
                    
                    # check whether pixel index is within target image dimension
                    if x >= 0 and x < W_t and y >= 0 and y < H_t:
                    
                        # add pixel value from source image to target image
                        im1[b, :, y, x] += im0[b, :, h, w]
                        
        return im1

    @staticmethod
    def backward(grad_output, im0, flow):
    
    
        # define gradient image in source domain
        im0_grad = torch.zeros_like(im0)
        
        # define gradient of flow
        #flow_grad = torch.zeros_like([B_s, H_s, W_s, 2])
        flow_grad = torch.zeros_like(flow)
        
        
        B_s = im0.shape[0]
        C_s = im0.shape[1]
        H_s = im0.shape[2]
        W_s = im0.shape[3]
        
        B_t = grad_output.shape[0]
        C_t = grad_output.shape[1]
        H_t = grad_output.shape[2]
        W_t = grad_output.shape[3]
        
        round_flow = torch.round(flow)
        

        # iterate over very pixel in source image and check whether it refers to some pixel in target image
        for b in range(B_s):
            for h in range(H_s):
                for w in range(W_s):
                    
                    # get pixel position in target image
                    x = int(round_flow[b, h, w, 0])
                    y = int(round_flow[b, h, w, 1])
                    
                    # check if forward-warp point to valid pixel in target image
                    if x >= 0 and x < W_t and y >= 0 and y < H_t:
                        # backpropagate gradient from previous layer through warping operation
                        im0_grad[b, :, h, w] = grad_output[b, :, y, x]
                        
                        # gradient of flow is 1 for layers connecting source to target image
                        flow_grad[b, h, w, 0] = 1
                        flow_grad[b, h, w, 1] = 1
        
        # first argument has to be
        # im0_grad = (delta im1) / (delta im0)
        
        # second argument has to be
        # flow_grad = (delta im1) / (delta grad)
        return im0_grad, flow_grad
