#ifndef LSTM_HPP
#define LSTM_HPP

#include <torch/torch.h>
#include <cublas_v2.h>

class LSTMImpl : public torch::nn::Module {
public:
    LSTMImpl(int64_t x, int64_t y, int64_t num_layers = 1, int64_t hidden_size = 512) : X(x), Y(y), num_layers_(num_layers), hidden_size_(hidden_size) {
        torch::nn::LSTMOptions option(x, hidden_size);
        option.num_layers(num_layers);
        lstm_ = register_module("lstm_", torch::nn::LSTM(option));
        final_layer_ = register_module("final_layer_", torch::nn::Linear(hidden_size, y));
        resetState();
    }

    torch::Tensor forward(const torch::Tensor& x) {
        //lstmは入力(input, (h_0, c_0))
        //inputのshapeは(seq_len, batch, input_size)
        //h_0, c_0は任意の引数で、状態を初期化できる
        //h_0, c_0のshapeは(num_layers_ * num_directions, batch, hidden_size_)
        //出力はoutput, (h_n, c_n)

        //実践的に入力は系列を1個ずつにバラしたものが入るのでshapeは(1, X)
        //まずそれを直す
        torch::Tensor input = x.view({ 1, 1, X });

        auto[output, h_and_c] = lstm_->forward(input, std::make_tuple(h_, c_));
        std::tie(h_, c_) = h_and_c;

        output = final_layer_->forward(output);

        return output;
    }

    void resetState() {
        h_ = torch::zeros({ num_layers_, 1, hidden_size_ });
        c_ = torch::zeros({ num_layers_, 1, hidden_size_ });
    }
private:
    int64_t X, Y, N, W, R;
    int64_t num_layers_;
    int64_t hidden_size_;
    torch::nn::LSTM lstm_{ nullptr };
    torch::nn::Linear final_layer_{ nullptr };
    torch::Tensor h_;
    torch::Tensor c_;
};
TORCH_MODULE(LSTM);

#endif //LSTM_HPP