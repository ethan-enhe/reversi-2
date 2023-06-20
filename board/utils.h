#ifndef UTILS_H
using ull=unsigned long long;
const int SIZE = 8;
const int _SIZE = SIZE + 1;

struct coor { // 坐标
    int x, y;
    coor operator+(const coor &b) const { return coor{x + b.x, y + b.y}; }
    bool inboard() { return x >= 1 && x <= SIZE && y >= 1 && y <= SIZE; }
};
#define get(v,c) (v)[(c).x][(c).y] // 直接用coor变量从二维数组中取值

// 八个方向
const coor pos8[] = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}, {1, 1}, {1, -1}, {-1, -1}, {-1, 1}};

#define UTILS_H
#endif
