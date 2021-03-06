#include "lstm.hpp"

LSTMImpl::LSTMImpl(int64_t input_size, int64_t output_size, int64_t num_layers, int64_t hidden_size) :
        input_size_(input_size),
        num_layers_(num_layers),
        hidden_size_(hidden_size) {
    torch::nn::LSTMOptions option(input_size, hidden_size);
    option.num_layers(num_layers);
    lstm_ = register_module("lstm_", torch::nn::LSTM(option));
    final_layer_ = register_module("final_layer_", torch::nn::Linear(hidden_size, output_size));
    resetState();
}

torch::Tensor LSTMImpl::forward(const torch::Tensor& x) {
    //lstmは入力(input, (h_0, c_0))
    //inputのshapeは(seq_len, batch, input_size)
    //h_0, c_0は任意の引数で、状態を初期化できる
    //h_0, c_0のshapeは(num_layers_ * num_directions, batch, hidden_size_)
    //出力はoutput, (h_n, c_n)

    //実践的に入力は系列を1個ずつにバラしたものが入るのでshapeは(1, input_size_)
    //まずそれを直す
    torch::Tensor input = x.view({ 1, 1, input_size_ });

    //outputのshapeは(seq_len, batch, num_directions * hidden_size)
    auto[output, h_and_c] = lstm_->forward(input, std::make_tuple(h_, c_));
    std::tie(h_, c_) = h_and_c;

    output = final_layer_->forward(output);

    return output;
}

void LSTMImpl::resetState() {
    h_ = torch::zeros({ num_layers_, 1, hidden_size_ });
    c_ = torch::zeros({ num_layers_, 1, hidden_size_ });
}

torch::Tensor LSTMImpl::forwardSequence(const torch::Tensor& input) {
    //lstmは入力(input, (h_0, c_0))
    //inputのshapeは(seq_len, batch, input_size)

    //outputのshapeは(seq_len, batch, num_directions * hidden_size)
    auto[output, h_and_c] = lstm_->forward(input);
    std::tie(h_, c_) = h_and_c;

    output = final_layer_->forward(output);

    return output;
}