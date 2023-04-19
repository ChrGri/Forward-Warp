#include <ATen/ATen.h>
#include <THC/THCAtomics.cuh>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#include "forward_warp.h"

//#include <torch/extension.h>

using at::native::detail::GridSamplerInterpolation;

static __forceinline__ __device__ 
int get_im_index(
    const int b,
    const int c,
    const int h,
    const int w,
    const size_t C,
    const size_t H,
    const size_t W) {
  return b*C*H*W + c*H*W + h*W + w;
}

template <typename scalar_t>
__global__ void forward_warp_cuda_forward_kernel(
    const int total_step,
    const scalar_t* im0,
    const scalar_t* flow,
    scalar_t* im1,
    const int B_s,
    const int C_s,
    const int H_s,
    const int W_s,
	const int B_t,
    const int C_t,
    const int H_t,
    const int W_t
	) 
	{
	// CUDA_KERNEL_LOOP(index, total_step-1) {
	// bug fix, thx to @tkkcc
	CUDA_KERNEL_LOOP(index, total_step) 
	{
		const int b_s = index / (H_s * W_s);
		const int h_s = (index-b_s*H_s*W_s) / W_s;
		const int w_s = index % W_s;
		const scalar_t x = flow[index*2+0];
		const scalar_t y = flow[index*2+1];
		 
		// get pixel position in target image
		const int x_nearest = static_cast<int>(::round(x));
		const int y_nearest = static_cast<int>(::round(y));
		
		// check whether pixel index is within target image dimension
		if(x_nearest>=0 && x_nearest<W_t && y_nearest>=0 && y_nearest<H_t)
		{
			// pointer to source image pixel
			const scalar_t* im0_p = im0 + get_im_index(b_s, 0, h_s, w_s, C_s, H_s, W_s);
			
			// pointer to target image pixel
			scalar_t* im1_p = im1 + get_im_index(b_s, 0, y_nearest, x_nearest, C_t, H_t, W_t);
			
			for (int c = 0; c < C_s; ++c, im0_p += H_s*W_s, im1_p += H_t*W_t) 
			{
				//  add pixel value from source image to target image
				atomicAdd(im1_p, *im0_p);
			}
		}
    
	}
}

template <typename scalar_t>
__global__ void forward_warp_cuda_backward_kernel(
    const int total_step,
    const scalar_t* grad_output,
    const scalar_t* im0,
    const scalar_t* flow,
    scalar_t* im0_grad,
    scalar_t* flow_grad,
    const int B_s,
    const int C_s,
    const int H_s,
    const int W_s,
	const int B_t,
    const int C_t,
    const int H_t,
    const int W_t
    ) 
	{
	CUDA_KERNEL_LOOP(index, total_step) {
    const int b_s = index / (H_s * W_s);
    const int h_s = (index-b_s*H_s*W_s) / W_s;
    const int w_s = index % W_s;
	
	// get pixel position in target image
    const scalar_t x = flow[index*2+0];
    const scalar_t y = flow[index*2+1];
	const int x_nearest = static_cast<int>(::round(x));
	const int y_nearest = static_cast<int>(::round(y));
	
	// check if forward-warp point to valid pixel in target image
	if(x_nearest>=0 && x_nearest<W_t && y_nearest>=0 && y_nearest<H_t)
	{
		// backpropagate gradient from previous layer through warping operation
		// get pointer of source image pixel
		scalar_t* im0_grad_p = im0_grad + get_im_index(b_s, 0, h_s, w_s, C_s, H_s, W_s);
		
		// get pointer of target image pixel
		const scalar_t* im1_grad_p = grad_output + get_im_index(b_s, 0, y_nearest, x_nearest, C_t, H_t, W_t);
		
		for (int c = 0; c < C_s; ++c, im0_grad_p += H_s*W_s, im1_grad_p += H_s*W_s) 
		{
			*im0_grad_p = *im1_grad_p;
			flow_grad[index*2+0] = 1;
			flow_grad[index*2+1] = 1;
		}
	}
  }
}

at::Tensor forward_warp_cuda_forward(
    const at::Tensor im0
    , const at::Tensor flow
	, const at::Tensor im1
	) 
{
	

	const int B_s = im0.size(0);
	const int C_s = im0.size(1);
	const int H_s = im0.size(2);
	const int W_s = im0.size(3);

	const int B_t = im1.size(0);
	const int C_t = im1.size(1);
	const int H_t = im1.size(2);
	const int W_t = im1.size(3);
	
	auto im1_out = at::zeros_like(im1);
	
	//auto im1 = at::zeros( (B_s, C_s, H_t, W_t) );
	//auto im1 = torch::zeros( (B_s, C_s, H_t, W_t) );
	//auto im1 = torch::ones({B_s, C_s, H_t, W_t}, torch::dtype(torch::kFloat64).requires_grad(true));
	


	const int total_step = B_s * H_s * W_s;


	AT_DISPATCH_FLOATING_TYPES(im0.scalar_type(), "forward_warp_forward_cuda", ([&] 
	{
	forward_warp_cuda_forward_kernel<scalar_t>
	<<<GET_BLOCKS(total_step), CUDA_NUM_THREADS>>>(
		  total_step,
		  im0.data<scalar_t>(),
		  flow.data<scalar_t>(),
		  im1_out.data<scalar_t>(),
		  B_s, C_s, H_s, W_s,
		  B_t, C_t, H_t, W_t);
	}));
	

	return im1_out;
}

std::vector<at::Tensor> forward_warp_cuda_backward(
    const at::Tensor grad_output
    , const at::Tensor im0
    , const at::Tensor flow
	) 
{
	auto im0_grad = at::zeros_like(im0);
	auto flow_grad = at::zeros_like(flow);
	
	const int B_s = im0.size(0);
	const int C_s = im0.size(1);
	const int H_s = im0.size(2);
	const int W_s = im0.size(3);
	
	const int B_t = grad_output.size(0);
	const int C_t = grad_output.size(1);
	const int H_t = grad_output.size(2);
	const int W_t = grad_output.size(3);
	
	
	const int total_step = B_s * H_s * W_s;
	
	
	

	AT_DISPATCH_FLOATING_TYPES(grad_output.type(), "forward_warp_backward_cuda", ([&] 
	{
    forward_warp_cuda_backward_kernel<scalar_t>
    <<<GET_BLOCKS(total_step), CUDA_NUM_THREADS>>>(
      total_step,
      grad_output.data<scalar_t>(),
      im0.data_ptr<scalar_t>(),
      flow.data<scalar_t>(),
      im0_grad.data<scalar_t>(),
      flow_grad.data<scalar_t>(),
      B_s, C_s, H_s, W_s,
	  B_t, C_t, H_t, W_t);
	}));

	return {im0_grad, flow_grad};
}
