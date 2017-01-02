#include <vector>

#include "caffe/layers/external_projection_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ExternalProjectionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  T_ = this->layer_param_.extproj_param().truncate();
  N_ = bottom[0]->shape(0);
  M_ = bottom[0]->shape(1);
  K_ = bottom[1]->shape(1);
}

//bottom[0]: N_ x M_
//bottom[1]: M_ x K_
//top[0]: N_ x K_
template <typename Dtype>
void ExternalProjectionLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->shape(1), bottom[1]->shape(0));
  vector<int> top_shape = bottom[0]->shape();//N_, M_
  top_shape[1] = K_;//N_, K_
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void ExternalProjectionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  if (bottom.size() == 1) { return; }
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* weight = bottom[1]->cpu_data();
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
      N_, K_, M_, (Dtype)1.,
      bottom_data, weight, (Dtype)0., top_data);
}

template <typename Dtype>
void ExternalProjectionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* weight = bottom[1]->cpu_data();
    // Gradient with respect to bottom data
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
        N_, M_, K_,
        (Dtype)1., top_diff, weight,
        (Dtype)0., bottom[0]->mutable_cpu_diff());
  }
}

#ifdef CPU_ONLY
STUB_GPU(ExternalProjectionLayer);
#endif

INSTANTIATE_CLASS(ExternalProjectionLayer);
REGISTER_LAYER_CLASS(ExternalProjection);

}  // namespace caffe
