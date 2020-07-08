#include "task.hpp"
#include <string>
#include <iostream>
#include <map>
#include <functional>

int main() {
    std::string task_name;
    std::cout << "タスク名: ";
    std::cin >> task_name;

    std::map<std::string, std::function<void()>> mp = {
            { "echo", echoTask },
            { "tree", treeTask },
            { "treeBatch", treeTaskBatch }
    };

    for (const auto&[name, func] : mp) {
        if (name == task_name) {
            func();
            return 0;
        }
    }

    std::cout << "Error : 実装されているタスク一覧" << std::endl;
    for (const auto& p : mp) {
        std::cout << p.first << std::endl;
    }
}