#include "../board/board.h"

#include <iostream>
using namespace std;
int main() {
    zobrist_hash::init();
    board x, y;
    x.putchess({4, 3}, 1, 1); // 1
    x.prt();
    y = x;
    y.putchess({5, 3}, -1, 1);
    y.prt();
    x.prt();
    return 0;
}
