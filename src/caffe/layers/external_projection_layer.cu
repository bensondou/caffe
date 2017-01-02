#include <vector>

#include "caffe/layers/external_projection_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ExternalProjectionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* weight = bottom[1]->gpu_data();
  if (N_ == 1) {
    caffe_gpu_gemv<Dtype>(CblasTrans, K_, M_, (Dtype)1.,
                         weight, bottom_data, (Dtype)0., top_data);
  } else {
    caffe_gpu_gemm<Dtype>(CblasNoTrans,
                          CblasNoTrans,
                          N_, K_, M_, (Dtype)1.,
                          bottom_data, weight, (Dtype)0., top_data);
  }
}

template <typename Dtype>
void ExternalProjectionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* weight = bottom[1]->gpu_data();
    // Gradient with respect to bottom data
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
          N_, M_, K_,
          (Dtype)1., top_diff, weight,
          (Dtype)0., bottom[0]->mutable_gpu_diff());
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ExternalProjectionLayer);

}  // namespace caffe
