#ifndef DIFFERENTIAL_NEURAL_COMPUTER_HPP
#define DIFFERENTIAL_NEURAL_COMPUTER_HPP

#include <torch/torch.h>

#define USE_LSTM_CONTROLLER

class ControllerImpl : public torch::nn::Module {
public:
    ControllerImpl(int64_t d_in, int64_t d_out, int64_t num_layers = 1, int64_t hidden_size = 512);
    torch::Tensor forward(const torch::Tensor& x);
    void resetState();
private:
#ifdef USE_LSTM_CONTROLLER
    torch::nn::LSTM l1{ nullptr };
    int64_t num_layers_;
    int64_t hidden_size_;
    torch::Tensor h_;
    torch::Tensor c_;
#else
    torch::nn::Linear l1{ nullptr };
#endif
    torch::nn::Linear l2{ nullptr };
};
TORCH_MODULE(Controller);

class DNCImpl : public torch::nn::Module {
public:
    DNCImpl(int64_t x, int64_t y, int64_t n, int64_t w, int64_t r);
    torch::Tensor forward(const torch::Tensor& x);
    static torch::Tensor C(const torch::Tensor& M, const torch::Tensor& k, const torch::Tensor& beta);
    static torch::Tensor u2a(const torch::Tensor& u);
    void resetState();
private:
    int64_t X, Y, N, W, R;
    Controller controller{ nullptr };
    torch::nn::Linear l_Wy{ nullptr };
    torch::nn::Linear l_Wxi{ nullptr };
    torch::nn::Linear l_Wr{ nullptr };
    torch::Tensor u, p, L, M, r, wr, ww;
};
TORCH_MODULE(DNC);

#endif //DIFFERENTIAL_NEURAL_COMPUTER_HPP
