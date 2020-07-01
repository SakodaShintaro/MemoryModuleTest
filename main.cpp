#include <iostream>
#include <vector>
#include <random>
#include <torch/torch.h>
#include "timer.hpp"
#include "differential_neural_computer.hpp"
#include "lstm.hpp"

//どっちかを使う
#define USE_LSTM
//#define USE_DNC

std::vector<float> onehot(int64_t x, int64_t n) {
    std::vector<float> ret(n, 0);
    ret[x] = 1.0;
    return ret;
}

int main() {
    constexpr int64_t X = 11;
    constexpr int64_t Y = X;

#ifdef USE_LSTM
    LSTM model(X, Y);
#elif defined(USE_DNC)
    constexpr int64_t N = 10;
    constexpr int64_t W = 10;
    constexpr int64_t R = 2;

    DNC model(X, Y, N, W, R);
#endif

    //Optimizerの準備
//    torch::optim::SGDOptions sgd_option(0.1);
//    sgd_option.momentum(0.9);
//    torch::optim::SGD optimizer(model->parameters(), sgd_option);
    torch::optim::Adam optimizer(model->parameters());

    constexpr int64_t DATA_NUM = 40000;
    constexpr int64_t INTERVAL = DATA_NUM / 50;
    float curr_loss = 0.0, curr_acc = 0.0;

    std::mt19937_64 engine(std::random_device{}());
    std::uniform_int_distribution<int64_t> dist_len(4, 7);
    std::uniform_int_distribution<int64_t> dist_X(0, X - 2);

    Timer timer;
    timer.start();

    std::ofstream learn_log("learn_log.txt");
    std::cout << std::fixed;
    learn_log << std::fixed;
    std::cout << "経過時間\t学習データ数\t損失\t精度" << std::endl;
    learn_log << "経過時間\t学習データ数\t損失\t精度" << std::endl;

    for (int64_t data_cnt = 1; data_cnt <= DATA_NUM; data_cnt++) {
        int64_t content_len = dist_len(engine);
        std::vector<int64_t> content;
        for (int64_t j = 0; j < content_len; j++) {
            content.push_back(dist_X(engine));
        }

        int64_t seq_len = content_len * 2;
        std::vector<std::vector<float>> x_seq_list(seq_len), t_seq_list(seq_len);
        for (int64_t i = 0; i < seq_len; i++) {
            if (i < content_len) {
                x_seq_list[i] = onehot(content[i], X);
            } else if (i == content_len) {
                x_seq_list[i] = onehot(X - 1, X);
            } else {
                x_seq_list[i].assign(X, 0);
            }

            if (i >= content_len) {
                t_seq_list[i] = onehot(content[i - content_len], X);
            }
        }

        model->resetState();
        std::vector<torch::Tensor> losses, accuracies;

        std::vector<int64_t> infer;
        for (int64_t i = 0; i < seq_len; i++) {
            torch::Tensor x = torch::tensor(x_seq_list[i]).view({ 1, X });
            torch::Tensor y = model->forward(x);

            if (i >= content_len) {
                torch::Tensor t = torch::tensor(t_seq_list[i]).view({ 1, Y });

                infer.push_back(torch::argmax(y).item<int64_t>());

                torch::Tensor loss_tensor = torch::sum(-t * torch::log_softmax(y, -1).view_as(t));
                losses.push_back(loss_tensor);

                torch::Tensor accuracy_tensor = (torch::argmax(y) == torch::argmax(t)).toType(torch::kFloat32);
                accuracies.push_back(accuracy_tensor);
            }
        }

//        std::cout << data_cnt << "回目" << std::endl;
//        std::cout << "推論: ";
//        for (int64_t i = 0; i < content_len; i++) {
//            std::cout << infer[i] << " \n"[i == content_len - 1];
//        }
//        std::cout << "正答: ";
//        for (int64_t i = 0; i < content_len; i++) {
//            std::cout << content[i] << " \n"[i == content_len - 1];
//        }

        torch::Tensor loss = torch::stack(losses).mean();
        torch::Tensor accuracy = torch::stack(accuracies).mean();
        std::cout << timer.elapsedTimeStr() << "\t" << std::setw(std::to_string(DATA_NUM).size()) << data_cnt << "\t" << loss.item<float>() << "\t" << accuracy.item<float>() << "\r" << std::flush;

        curr_loss += loss.item<float>();
        curr_acc += accuracy.item<float>();

        if (data_cnt % INTERVAL == 0) {
            curr_loss /= INTERVAL;
            curr_acc /= INTERVAL;
            std::cout << timer.elapsedTimeStr() << "\t" << std::setw(std::to_string(DATA_NUM).size()) << data_cnt << "\t" << curr_loss << "\t" << curr_acc << std::endl;
            learn_log << timer.elapsedTimeStr() << "\t" << std::setw(std::to_string(DATA_NUM).size()) << data_cnt << "\t" << curr_loss << "\t" << curr_acc << std::endl;
            curr_loss = 0;
            curr_acc = 0;
        }

        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
    }
}