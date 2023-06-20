#ifndef BOARD_H
#include <cstring>
#include <iostream>
#include <random>

#include "./utils.h"

namespace zobrist_hash { // zobrist hashing,需要先调用init初始化哈希表
ull h[_SIZE][_SIZE][3];
void init() {
    std::mt19937_64 rnd(114514);
    for (int i = 1; i <= SIZE; i++)
        for (int j = 1; j <= SIZE; j++) {
            h[i][j][0] = rnd();
            h[i][j][2] = rnd();
        }
}
} // namespace zobrist_hash

namespace board_class {

struct board {
    // 使用char存储,省空间
    char b[_SIZE][_SIZE]; // 棋盘,空0,先手1,后手-1
    char turn;            // 最后一个走棋的人,默认-1
    ull h;

    board() {
        h = 0;
        turn = -1;
        memset(b, 0, sizeof(b));
        change({4, 4}, -1);
        change({4, 5}, 1);
        change({5, 4}, 1);
        change({5, 5}, -1);
    }

    void prt() {
        std::cout << "hash: " << h << std::endl;
        std::cout << "lastput: " << (turn == 1 ? "O " : "X ") << std::endl;
        for (int i = 1; i <= SIZE; i++) {
            for (int j = 1; j <= SIZE; j++) std::cout << (b[i][j] ? (b[i][j] == 1 ? "O " : "X ") : "  ");
            std::cout << std::endl;
        }
    }
    void change(coor c, char v) { // 修改,因为后面可能还要维护hash,所以单独搞一个函数
        ull *_h = get(zobrist_hash::h, c);
        h ^= _h[get(b, c) + 1] ^ (v + 1);
        get(b, c) = v;
    }
    /*
     * 传入(坐标,下棋方,是否真的下)
     * 返回能否在这里下,如果可以,并且第三个参数为1,就真的下一步
     */
    bool putchess(coor c, char p, bool f = 0) {
        if (!c.inboard() || get(b, c)) return 0;
        bool res = 0;
        for (int i = 0; i < 8; i++) {
            bool cnt = 0, canflip = 0;
            for (coor j = c + pos8[i]; j.inboard() && get(b, j); j = j + pos8[i]) {
                if (get(b, j) == p) {
                    if (cnt) {
                        res = canflip = 1;
                        if (!f) return 1;
                    }
                    break;
                } else
                    cnt = 1;
            }
            if (f && canflip)
                for (coor j = c + pos8[i]; j.inboard() && get(b, j) != p; j = j + pos8[i]) change(j, p);
        }
        if (f && res) change(c, p), turn = p;
        return res;
    }
    bool canput(char p) { // p是否有合法的棋步
        for (int i = 1; i <= SIZE; i++)
            for (int j = 1; j <= SIZE; j++) {
                if (putchess({i, j}, p)) return 1;
            }
        return 0;
    }
    char win() { //-1,0,1表示后手胜,平,先手胜,2表示没结束
        if (canput(1) || canput(-1)) return 2;
        int c1 = 0, c_1 = 0;
        for (int i = 1; i <= SIZE; i++)
            for (int j = 1; j <= SIZE; j++) {
                if (b[i][j] == 1) ++c1;
                if (b[i][j] == -1) ++c_1;
            }
        return (c1 == c_1 ? 0 : (c1 > c_1 ? 1 : -1));
    }
};
} // namespace board_class
using board_class::board;

#define BOARD_H
#endif
