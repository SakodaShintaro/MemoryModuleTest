#ifndef DIFFERENTIAL_NEURAL_COMPUTER_HPP
#define DIFFERENTIAL_NEURAL_COMPUTER_HPP

#include <torch/torch.h>

class ControllerImpl : public torch::nn::Module {
public:
    ControllerImpl(int64_t d_in, int64_t d_out) {
//        l1 = register_module("l1", torch::nn::LSTM(d_in, d_out));
        constexpr int64_t hidden_dim = 512;
        l1 = register_module("l1", torch::nn::Linear(d_in, hidden_dim));
        l2 = register_module("l2", torch::nn::Linear(hidden_dim, d_out));
    }
    torch::Tensor forward(const torch::Tensor& x) {
//        auto result = l1->forward(x);
//        auto&[out1, out2] = result;
//        return l2->forward(out1);
        return l2->forward(l1->forward(x));
    }
    void resetState() {
    }
private:
//    torch::nn::LSTM l1{ nullptr };
    torch::nn::Linear l1{ nullptr };
    torch::nn::Linear l2{ nullptr };
};
TORCH_MODULE(Controller);

class DNCImpl : public torch::nn::Module {
public:
    DNCImpl(int64_t x, int64_t y, int64_t n, int64_t w, int64_t r) : X(x), Y(y), N(n), W(w), R(r) {
        const int64_t h = W * R + 3 * W + 5 * R + 3;
        controller = register_module("controller", Controller(W * R + X, Y + h));
        l_Wy = register_module("l_Wy", torch::nn::Linear(Y + h, Y));
        l_Wxi = register_module("l_Wxi", torch::nn::Linear(Y + h, h));
        l_Wr = register_module("l_Wr", torch::nn::Linear(R * W, Y));
        resetState();
    }

    torch::Tensor forward(const torch::Tensor& x) {
        //x = (1, X), r = (1, R * W)
        torch::Tensor chi = torch::cat({ x, r }, 1);
        chi = chi.view({ 1, 1, X + R * W });

        //out = (1, 1, Y + W * R + 3 * W + 5 * R + 3)
        torch::Tensor out = controller->forward(chi);

        //ここで余計な次元を削減
        //out = (1, Y + W * R + 3 * W + 5 * R + 3)
        out = out.view({ 1, -1 });

        //v = (1, Y)
        torch::Tensor v = l_Wy->forward(out);

        //xi = (1, W * R + 3 * W + 5 * R + 3)
        torch::Tensor xi = l_Wxi->forward(out);

        std::vector<torch::Tensor> split = torch::split_with_sizes(xi, { W * R, R, W, 1, W, W, R, 1, 1, 3 * R }, 1);
        assert(split.size() == 10);
        torch::Tensor kr = split[0];
        torch::Tensor beta_r = split[1];
        torch::Tensor kw = split[2];
        torch::Tensor beta_w = split[3];
        torch::Tensor e = split[4];
        torch::Tensor nu = split[5];
        torch::Tensor f = split[6];
        torch::Tensor ga = split[7];
        torch::Tensor gw = split[8];
        torch::Tensor pi = split[9];

        //kr = (R, W)
        kr = kr.view({ R, W });
        beta_r = 1 + torch::softplus(beta_r);
        beta_w = 1 + torch::softplus(beta_w);
        e = torch::sigmoid(e);
        f = torch::sigmoid(f);
        ga = torch::sigmoid(ga);
        gw = torch::sigmoid(gw);
        pi = torch::softmax(pi.view({ R, 3 }), 1);

        torch::Tensor psi_mat = 1 - torch::matmul(torch::ones({ N, 1 }), f) * wr;
        torch::Tensor psi = torch::ones({ N, 1 });
        for (int64_t i = 0; i < R; i++) {
            psi = psi * psi_mat.slice(1, i, i + 1).view({ N, 1 });
        }

        u = (u + ww - (u * ww)) * psi;

        torch::Tensor a = u2a(u).view({ N, 1 });
        torch::Tensor cw = C(M, kw, beta_w);
        ww = torch::matmul(torch::matmul(a, ga) + torch::matmul(cw, 1.0 - ga), gw);

        //Write Memory
        M = M * (torch::ones({ N, W }) - torch::matmul(ww, e)) + torch::matmul(ww, nu);

        p = (1.0 - torch::matmul(torch::ones({ N, 1 }), torch::sum(ww).view({ 1, 1 }))) * p + ww;
        torch::Tensor ww_mat = torch::matmul(ww, torch::ones({ 1, N }));
        L = (1.0 - ww_mat - torch::t(ww_mat)) * L + torch::matmul(ww, torch::t(p));
        L = L * (torch::ones({ N, N }) - torch::eye(N));

        torch::Tensor fw = torch::matmul(L, wr);
        torch::Tensor bw = torch::matmul(torch::t(L), wr);

        std::vector<torch::Tensor> cr_list;
        for (int64_t i = 0; i < R; i++) {
            cr_list.push_back(C(M, kr[i].view({ 1, W }), beta_r[0][i].view({ 1, 1 })));
        }

        torch::Tensor cr = torch::cat(cr_list);

        torch::Tensor bacrfo = torch::cat({
                                                  torch::t(bw).view({ R, N, 1 }),
                                                  torch::t(cr).view({ R, N, 1 }),
                                                  torch::t(fw).view({ R, N, 1 }),
                                          }, 2);
        pi = pi.view({ R, 3, 1 });
        wr = torch::t(torch::bmm(bacrfo, pi).view({ R, N }));

        //read from memory, r = (1, R * W)
        r = torch::matmul(torch::t(M), wr).view({ 1, R * W });

        return l_Wr->forward(r) + v;
    }

    static torch::Tensor C(const torch::Tensor& M, const torch::Tensor& k, const torch::Tensor& beta) {
        std::vector<torch::Tensor> ret_list(M.size(0));
        for (int64_t i = 0; i < M.size(0); i++) {
            ret_list[i] = torch::cosine_similarity(M[i].view_as(k), k) * beta;
        }

        return torch::softmax(torch::cat(ret_list, 0), 0);
    }

    static torch::Tensor u2a(const torch::Tensor& u) {
        int64_t N = u.size(0);
        std::vector<std::pair<float, int64_t>> u_value;
        for (int64_t i = 0; i < N; i++) {
            u_value.emplace_back(u[i].item<float>(), i);
        }

        std::sort(u_value.begin(), u_value.end());

        std::vector<float> a_list(N);
        float cum_prod = 1.0;
        for (int64_t i = 0; i < N; i++) {
            a_list[u_value[i].second] = (cum_prod * (1.0 - u_value[i].first));
            cum_prod *= u_value[i].first;
        }

        return torch::tensor(a_list);
    }

    void resetState() {
        controller->resetState();
        u = torch::zeros({ N, 1 });
        p = torch::zeros({ N, 1 });
        L = torch::zeros({ N, N });
        M = torch::zeros({ N, W });
        r = torch::zeros({ 1, R * W });
        wr = torch::zeros({ N, R });
        ww = torch::zeros({ N, 1 });
    }
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
