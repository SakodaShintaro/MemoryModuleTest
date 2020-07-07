#include "task.hpp"

#include <iostream>
#include <stack>
#include <random>
#include <torch/torch.h>
#include "timer.hpp"
#include "double_ostream.hpp"
#include "differential_neural_computer.hpp"
#include "lstm.hpp"

//どっちかを使う
#define USE_LSTM
//#define USE_DNC

//隣接リストの形式で表現
using Tree = std::vector<std::vector<int64_t>>;

//木の構築(辺の構成を返す)
Tree makeTree(int64_t N) {
    static std::mt19937_64 engine(std::random_device{}());

    Tree t(N);

    for (int64_t i = 1; i < N; i++) {
        std::uniform_int_distribution<int64_t> dist(0, i - 1);
        int64_t n = dist(engine);
        t[i].push_back(n);
        t[n].push_back(i);
    }
    return t;
}

void dfsTraverse(int64_t curr_node, int64_t pre_node, const Tree& tree, std::vector<int64_t>& result) {
    //自分を記録
    result.push_back(curr_node);

    for (int64_t next_node : tree[curr_node]) {
        if (next_node == pre_node) {
            continue;
        }

        //next_nodeに移動して記録
        dfsTraverse(next_node, curr_node, tree, result);

        //自分に戻ってくるので再度自分を記録
        result.push_back(curr_node);
    }
}

std::vector<int64_t> dfs(const Tree& tree, int64_t node) {
    std::vector<int64_t> result;
    dfsTraverse(node, -1, tree, result);
    return result;
}

std::vector<int64_t> bfs(const Tree& tree, int64_t node) {
    std::vector<int64_t> result;
    std::queue<std::pair<int64_t, int64_t>> q;
    q.push(std::make_pair(node, -1));
    while (!q.empty()) {
        std::pair<int64_t, int64_t> top = q.front();
        q.pop();
        result.push_back(top.first);

        for (int64_t next_node : tree[top.first]) {
            if (next_node == top.second) {
                continue;
            }

            q.push(std::make_pair(next_node, top.first));
        }
    }

    return result;
}

void treeTask() {
    constexpr int64_t MAX_NODE_NUM = 6;
    constexpr int64_t INPUT_DIM = MAX_NODE_NUM + 1;
    constexpr int64_t Y = INPUT_DIM;

#ifdef USE_LSTM
    LSTM model(INPUT_DIM, Y);
#elif defined(USE_DNC)
    constexpr int64_t N = 10;
    constexpr int64_t W = 10;
    constexpr int64_t R = 2;

    DNC model(INPUT_DIM, Y, N, W, R);
#endif

    //Optimizerの準備
//    torch::optim::SGDOptions sgd_option(0.1);
//    sgd_option.momentum(0.9);
//    torch::optim::SGD optimizer(model->parameters(), sgd_option);
    torch::optim::Adam optimizer(model->parameters());

    constexpr int64_t DATA_NUM = 50000;
    constexpr int64_t INTERVAL = DATA_NUM / 25;
    float curr_loss = 0.0, curr_acc = 0.0, curr_perf_acc = 0.0;

    std::mt19937_64 engine(std::random_device{}());
    std::uniform_int_distribution<int64_t> dist_node_num(MAX_NODE_NUM, MAX_NODE_NUM);

    Timer timer;
    timer.start();

    std::ofstream learn_log("learn_log.txt");
    DoubleOstream ost(std::cout, learn_log);
    ost << "経過時間\t学習データ数\t損失\t精度\t完全一致精度" << std::endl << std::fixed;

    for (int64_t data_cnt = 1; data_cnt <= DATA_NUM; data_cnt++) {
        int64_t node_num = dist_node_num(engine);
        Tree tree = makeTree(node_num);

        //rootをランダムに選択し、dfs, bfsの結果を取得
        std::uniform_int_distribution<int64_t> dist_root(0, node_num - 1);
        int64_t root = dist_root(engine);
        std::vector<int64_t> dfs_result = dfs(tree, root);
        std::vector<int64_t> bfs_result = bfs(tree, root);

        int64_t input_len = dfs_result.size();

        int64_t seq_len = (input_len + node_num);

        std::vector<std::vector<float>> input, teacher(seq_len);
        for (int64_t i = 0; i < seq_len; i++) {
            if (i < input_len) {
                input.push_back(onehot(dfs_result[i], INPUT_DIM));
            } else if (i == input_len) {
                input.push_back(onehot(MAX_NODE_NUM, INPUT_DIM));
            } else {
                input.emplace_back(INPUT_DIM, 0);
            }

            if (i >= input_len) {
                teacher[i] = onehot(bfs_result[i - dfs_result.size()], INPUT_DIM);
            }
        }

        model->resetState();
        std::vector<torch::Tensor> losses, accuracies;

        std::vector<int64_t> infer;
        for (int64_t i = 0; i < seq_len; i++) {
            torch::Tensor x = torch::tensor(input[i]).view({ 1, INPUT_DIM });
            torch::Tensor y = model->forward(x);

            if (i >= input_len) {
                torch::Tensor t = torch::tensor(teacher[i]).view({ 1, Y });

                infer.push_back(torch::argmax(y).item<int64_t>());

                torch::Tensor loss_tensor = torch::sum(-t * torch::log_softmax(y, -1).view_as(t));
                losses.push_back(loss_tensor);

                torch::Tensor accuracy_tensor = (torch::argmax(y) == torch::argmax(t)).toType(torch::kFloat32);
                accuracies.push_back(accuracy_tensor);
            }
        }

        torch::Tensor loss = torch::stack(losses).mean();
        torch::Tensor accuracy = torch::stack(accuracies).mean();
        float perf_acc = (infer == bfs_result);
        std::cout << timer.elapsedTimeStr() << "\t"
                  << std::setw(std::to_string(DATA_NUM).size()) << data_cnt << "\t"
                  << loss.item<float>() << "\t"
                  << accuracy.item<float>() << "\t"
                  << perf_acc << "\r" << std::flush;

        curr_loss += loss.item<float>();
        curr_acc += accuracy.item<float>();
        curr_perf_acc += perf_acc;

        if (data_cnt % INTERVAL == 0) {
            curr_loss /= INTERVAL;
            curr_acc /= INTERVAL;
            curr_perf_acc /= INTERVAL;
            ost << timer.elapsedTimeStr() << "\t"
                << std::setw(std::to_string(DATA_NUM).size()) << data_cnt << "\t"
                << curr_loss << "\t"
                << curr_acc << "\t"
                << curr_perf_acc << std::endl;
            curr_loss = 0;
            curr_acc = 0;
            curr_perf_acc = 0;
        }

        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
    }
}

void treeTaskBatch() {
    constexpr int64_t MAX_NODE_NUM = 6;
    constexpr int64_t INPUT_DIM = MAX_NODE_NUM + 1;
    constexpr int64_t Y = INPUT_DIM;

#ifdef USE_LSTM
    LSTM model(INPUT_DIM, Y);
#elif defined(USE_DNC)
    constexpr int64_t N = 10;
    constexpr int64_t W = 10;
    constexpr int64_t R = 2;

    DNC model(INPUT_DIM, Y, N, W, R);
#endif

    //Optimizerの準備
//    torch::optim::SGDOptions sgd_option(0.1);
//    sgd_option.momentum(0.9);
//    torch::optim::SGD optimizer(model->parameters(), sgd_option);
    torch::optim::Adam optimizer(model->parameters());

    constexpr int64_t STEP_NUM = 50000;
    constexpr int64_t BATCH_SIZE = 4;
    constexpr int64_t INTERVAL = STEP_NUM / 25;
    float curr_loss = 0.0, curr_acc = 0.0, curr_perf_acc = 0.0;

    std::mt19937_64 engine(std::random_device{}());
    std::uniform_int_distribution<int64_t> dist_node_num(MAX_NODE_NUM, MAX_NODE_NUM);

    Timer timer;
    timer.start();

    std::ofstream learn_log("learn_log.txt");
    DoubleOstream ost(std::cout, learn_log);
    ost << "経過時間\t学習データ数\t損失\t精度\t完全一致精度" << std::endl << std::fixed;

    for (int64_t step = 1; step <= STEP_NUM; step++) {
        int64_t node_num = dist_node_num(engine);
        int64_t input_len = 2 * (node_num - 1);
        int64_t seq_len = (input_len + node_num);

        std::vector<float> input, teacher;
        for (int64_t b = 0; b < BATCH_SIZE; b++) {
            //木をランダムに構築
            Tree tree = makeTree(node_num);

            //rootをランダムに選択し、dfs, bfsの結果を取得
            std::uniform_int_distribution<int64_t> dist_root(0, node_num - 1);
            int64_t root = dist_root(engine);
            std::vector<int64_t> dfs_result = dfs(tree, root);
            std::vector<int64_t> bfs_result = bfs(tree, root);
            assert(dfs_result.size() == (uint64_t)input_len);
            assert(bfs_result.size() == (uint64_t)node_num);

            //入力の構築
            for (int64_t i = 0; i < seq_len; i++) {
                std::vector<float> add;
                if (i < input_len) {
                    add = onehot(dfs_result[i], INPUT_DIM);
                } else if (i == input_len) {
                    add = onehot(MAX_NODE_NUM, INPUT_DIM);
                } else {
                    add.assign(INPUT_DIM, 0);
                }
                input.insert(input.end(), add.begin(), add.end());
            }

            //教師の構築
            for (int64_t i = 0; i < node_num; i++) {
                std::vector<float> add = onehot(bfs_result[i], INPUT_DIM);
                teacher.insert(teacher.end(), add.begin(), add.end());
            }
        }

        torch::Tensor input_tensor = torch::tensor(input);
        torch::Tensor teacher_tensor = torch::tensor(teacher);

        //(seq_len, batch, output_size)
        torch::Tensor output_without_softmax = model->forwardSequence(input_tensor);

        //assert(false);
        //ここでスライス
        torch::Tensor sliced_output = output_without_softmax;

        //損失計算(方向とかに注意)
        torch::Tensor loss = torch::sum(-teacher_tensor * torch::log_softmax(sliced_output, -1));
        torch::Tensor accuracy = loss;
        torch::Tensor perfect_acc = accuracy;

        std::cout << timer.elapsedTimeStr() << "\t"
                  << std::setw(std::to_string(STEP_NUM).size()) << step << "\t"
                  << loss.item<float>() << "\t"
                  << accuracy.item<float>() << "\t"
                  << perfect_acc.item<float>() << "\r" << std::flush;

        curr_loss += loss.item<float>();
        curr_acc += accuracy.item<float>();
        curr_perf_acc += perfect_acc.item<float>();

        if (step % INTERVAL == 0) {
            curr_loss /= INTERVAL;
            curr_acc /= INTERVAL;
            curr_perf_acc /= INTERVAL;
            ost << timer.elapsedTimeStr() << "\t"
                << std::setw(std::to_string(STEP_NUM).size()) << step << "\t"
                << curr_loss << "\t"
                << curr_acc << "\t"
                << curr_perf_acc << std::endl;
            curr_loss = 0;
            curr_acc = 0;
            curr_perf_acc = 0;
        }

        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
    }
}