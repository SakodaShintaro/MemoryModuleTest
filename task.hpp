#ifndef TASK_HPP
#define TASK_HPP

#include <vector>
#include <cstdint>

void treeTask();
void treeTaskBatch();

//---------------------------
//    共通して使う関数の記述
//---------------------------

//onehotベクトル化
static std::vector<float> onehot(int64_t x, int64_t n) {
    std::vector<float> ret(n, 0);
    ret[x] = 1.0;
    return ret;
}

#endif //TASK_HPP