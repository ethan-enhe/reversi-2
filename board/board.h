#ifndef BOARD_H
#include "./utils.h"

namespace board_class {
struct board {
    int8_t b[_SIZE][_SIZE]; // 棋盘,空0,先手1,后手-1
    int8_t turn;            // 最后一个走棋的人,默认-1
    /*
     * 传入(坐标,下棋方,是否真的下)
     * 返回能否在这里下,如果可以,并且第三个参数为1,就真的下一步
     */
    bool putchess(coor c, int8_t p, bool f = 0) {
        if (!c.inboard()) return 0;
        for(int i=0;i<8;i++){

        }
    }
};

} // namespace board_class

#define BOARD_H
#endif
