#ifndef LSTM_HPP
#define LSTM_HPP

#include <torch/torch.h>

class LSTMImpl : public torch::nn::Module {
public:
    LSTMImpl(int64_t input_size, int64_t output_size, int64_t num_layers = 1, int64_t hidden_size = 512);

    //入力を受けて1ステップ分LSTMの推論を進める関数:これを直に触って使うのはやめた方が良さそう
    torch::Tensor forward(const torch::Tensor& x);

    //内部状態をリセットする関数:上のforwardを使わないようにできれば自然とこの関数も不要になる
    void resetState();
private:
    int64_t input_size_;
    int64_t num_layers_;
    int64_t hidden_size_;
    torch::nn::LSTM lstm_{ nullptr };
    torch::nn::Linear final_layer_{ nullptr };
    torch::Tensor h_;
    torch::Tensor c_;
};
TORCH_MODULE(LSTM);

#endif //LSTM_HPP