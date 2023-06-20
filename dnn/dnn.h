#ifndef MICRODNN_H
#define MICRODNN_H

#define NDEBUG
#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")
#pragma GCC target("sse,sse2,sse3,ssse3,sse4.1,sse4.2,avx,avx2,bmi,bmi2,lzcnt,popcnt")

#include <bits/stdc++.h>

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

using namespace Eigen;
using namespace std;

using vec = VectorXf;
using mat = MatrixXf;
using ten0 = Tensor<float, 0>;
using ten1 = Tensor<float, 1>;
using ten2 = Tensor<float, 2>;
using ten3 = Tensor<float, 3>;
using ten4 = Tensor<float, 4>;
using vecmap = Map<vec>;
using matmap = Map<mat>;
using ten3map = TensorMap<ten3>;
using vec_batch = vector<vec>;
using batch = pair<vec_batch, vec_batch>;
const float INF = 1e8;
const float EPS = 1e-8;

//{{{ Random
mt19937_64 mr(chrono::system_clock::now().time_since_epoch().count());
// mt19937_64 mr(65);
float rd(float l, float r) { return uniform_real_distribution<float>(l, r)(mr); }
float nd(float l, float r) { return normal_distribution<float>(l, r)(mr); }
int ri(int l, int r) { return uniform_int_distribution<int>(l, r)(mr); }
//}}}
//{{{ Utils
vec make_vec(const vector<double> &x) {
    vec res;
    res.resize(x.size());
    for (int i = 0; i < (int)x.size(); i++) res(i) = x[i];
    return res;
}
template <typename T>
vecmap to_vecmap(T &x) {
    return vecmap(x.data(), x.size());
}
//}}}
//{{{ Error Function
float mylog(float x) {
    if (x == 0) return -INF;
    return log(x);
}
float variance(const vec_batch &out, const vec_batch &label) {
    auto square = [](float x) { return x * x; };
    const int batch_sz = out.size();
    float res = 0;
    for (int i = 0; i < batch_sz; i++)
        for (int j = 0; j < out[i].rows(); j++) res += square(out[i](j) - label[i](j));

    return res / batch_sz / out[0].rows();
}
float sqrtvariance(const vec_batch &out, const vec_batch &label) {
    auto square = [](float x) { return x * x; };
    const int batch_sz = out.size();
    float res = 0, ans = 0;
    for (int i = 0; i < batch_sz; i++) {
        res = 0;
        for (int j = 0; j < out[i].rows(); j++) res += square(out[i](j) - label[i](j));
        ans += sqrt(res / out[i].rows());
    }

    return ans / batch_sz;
}
// 分2类，交叉熵
float crossentropy_2(const vec_batch &out, const vec_batch &label) {
    const int batch_sz = out.size();
    float res = 0;
    for (int i = 0; i < batch_sz; i++)
        for (int j = 0; j < out[i].rows(); j++)
            res -= label[i](j) * mylog(out[i](j)) + (1. - label[i](j)) * mylog(1. - out[i](j));
    return res / batch_sz;
}
float crossentropy_k(const vec_batch &out, const vec_batch &label) {
    const int batch_sz = out.size();
    float res = 0;
    for (int i = 0; i < batch_sz; i++)
        for (int j = 0; j < out[i].rows(); j++) res -= label[i](j) * mylog(out[i](j));

    return res / batch_sz;
}
float chk_k(const vec_batch &out, const vec_batch &label) {
    const int batch_sz = out.size();
    float res = 0;
    for (int i = 0; i < batch_sz; i++) {
        int mx = 0;
        for (int j = 0; j < out[i].rows(); j++)
            if (out[i](j) > out[i](mx)) mx = j;
        res += (label[i](mx) > 0.9);
    }
    return -res / batch_sz;
}
float chk_2(const vec_batch &out, const vec_batch &label) {
    const int batch_sz = out.size();
    float res = 0;
    for (int i = 0; i < batch_sz; i++) {
        for (int j = 0; j < out[i].rows(); j++) res += (out[i](j) > 0.5) == (label[i](j) > 0.5);
    }
    return -res / batch_sz / out[0].rows();
}
//}}}
//{{{ Optimizer
struct optimizer {
    virtual void upd(vecmap, const vecmap &) = 0;
};

// 一个类，用来维护每一个系数矩阵对应的额外信息，如动量之类的
template <const int N>
struct optimizer_holder : public optimizer {
    unordered_map<const float *, vec> V[N];
    template <const int I>
    vec &get(const vecmap &x) {
        auto it = V[I].find(x.data());
        if (it != V[I].end())
            return it->second;
        else
            return V[I][x.data()] = vec::Zero(x.size());
    }
};

struct sgd : public optimizer_holder<0> {
    float lr, lambda;
    sgd(float lr = 0.01, float lambda = 0) : lr(lr), lambda(lambda) {}
    void upd(vecmap w, const vecmap &gw) { w -= (gw + w * lambda) * lr; }
};
struct nesterov : public optimizer_holder<1> {
    float alpha, mu, lambda;
    nesterov(float alpha = 0.01, float mu = 0.9, float lambda = 0) : alpha(alpha), mu(mu), lambda(lambda) {}
    void upd(vecmap w, const vecmap &gw) {
        // 原始版：
        // w'+=alpha*v
        // gw=grad(w')
        // v=alpha*v-gw*mu
        // w+=v
        // 修改版：
        // gw=grad(w)
        // w-=alpha*v;
        // v=alpha*b-gw*mu;
        // w+=v*(1+alpha);
        vec &v = get<0>(w);
        w -= mu * v;
        v = mu * v - (gw + lambda * w) * alpha;
        w += (1. + mu) * v;
    }
};

struct adam : public optimizer_holder<2> {
    float lr, rho1, rho2, eps, lambda;
    float mult1, mult2;
    adam(float lr = 0.001, float rho1 = 0.9, float rho2 = 0.999, float eps = 1e-6, float lambda = 0)
        : lr(lr), rho1(rho1), rho2(rho2), eps(eps), lambda(lambda), mult1(1), mult2(1) {}
    void upd(vecmap w, const vecmap &gw) {
        vec &s = get<0>(w), &r = get<1>(w);
        mult1 *= rho1, mult2 *= rho2;
        s = s * rho1 + (gw + w * lambda) * (1. - rho1);
        r = r * rho2 + (gw + w * lambda).cwiseAbs2() * (1. - rho2);
        w.array() -= lr * s.array() / (sqrt(r.array() / (1. - mult2) + eps)) / (1. - mult1);
    }
};

//}}}
//{{{ Layer
struct layer;
using layerp = shared_ptr<layer>;

// 一层神经元的基类
struct layer {
    string name;
    vec_batch out, grad;
    int batch_sz;
    bool train_mode;

    layer(const string &name) : name(name), batch_sz(0), train_mode(1) {}

    // 更改batch-size 和 trainmode 选项
    void set_train_mode(const bool &new_train_mode) { train_mode = new_train_mode; }
    virtual void _resize(int){};
    void resize(int new_batch_sz) {
        if (batch_sz != new_batch_sz) {
            batch_sz = new_batch_sz;
            out.resize(batch_sz);
            grad.resize(batch_sz);
            _resize(batch_sz);
        }
    }
    // 把输入向量运算后放到Z里
    virtual void forward(const vec_batch &in) {}
    // 根据输入向量，和答案对下一层神经元输入的偏导，算出答案对这一层神经元输入的偏导
    virtual void backward(const vec_batch &, const vec_batch &) = 0;

    virtual void clear_grad() {}
    // 返回这一层对应的系数，梯度矩阵的指针
    virtual void write(ostream &io) {}
    virtual void read(istream &io) {}
    // 以 rate 学习率梯度下降
    virtual void sgd(float) {}
    virtual void upd(optimizer &opt) {}
};

//}}}
//{{{ Activate Function
struct sigmoid : public layer {
    sigmoid() : layer("sigmoid") {}
    void forward(const vec_batch &in) {
        for (int i = 0; i < batch_sz; i++)
            out[i] = in[i].unaryExpr([](float x) -> float { return 1. / (exp(-x) + 1); });
    }
    void backward(const vec_batch &in, const vec_batch &nxt_grad) {
        for (int i = 0; i < batch_sz; i++)
            grad[i] = nxt_grad[i].cwiseProduct(out[i].unaryExpr([](float x) -> float { return x * (1. - x); }));
    }
};
struct th : public layer {
    th() : layer("tanh") {}
    void forward(const vec_batch &in) {
        for (int i = 0; i < batch_sz; i++) out[i] = in[i].unaryExpr([](float x) { return tanh(x); });
    }
    void backward(const vec_batch &in, const vec_batch &nxt_grad) {
        for (int i = 0; i < batch_sz; i++)
            grad[i] = nxt_grad[i].cwiseProduct(out[i].unaryExpr([](float x) -> float { return 1. - x * x; }));
    }
};
struct relu : public layer {
    relu() : layer("relu") {}
    void forward(const vec_batch &in) {
        for (int i = 0; i < batch_sz; i++) out[i] = in[i].unaryExpr([](float x) -> float { return max((float)0, x); });
    }
    void backward(const vec_batch &in, const vec_batch &nxt_grad) {
        for (int i = 0; i < batch_sz; i++)
            grad[i] = nxt_grad[i].cwiseProduct(in[i].unaryExpr([](float x) -> float { return x < 0 ? 0 : 1; }));
    }
};
struct hardswish : public layer {
    hardswish() : layer("hardswish") {}
    void forward(const vec_batch &in) {
        for (int i = 0; i < batch_sz; i++)
            out[i] = in[i].unaryExpr([](float x) -> float {
                if (x >= 3) return x;
                if (x <= -3) return 0;
                return x * (x + 3) / 6;
            });
    }
    void backward(const vec_batch &in, const vec_batch &nxt_grad) {
        for (int i = 0; i < batch_sz; i++)
            grad[i] = nxt_grad[i].cwiseProduct(in[i].unaryExpr([](float x) -> float {
                if (x >= 3) return 1;
                if (x <= -3) return 0;
                return x / 3 + 0.5;
            }));
    }
};
struct swish : public layer {
    swish() : layer("swish") {}
    void forward(const vec_batch &in) {
        for (int i = 0; i < batch_sz; i++) out[i] = in[i].unaryExpr([](float x) -> float { return x / (exp(-x) + 1); });
    }
    void backward(const vec_batch &in, const vec_batch &nxt_grad) {
        for (int i = 0; i < batch_sz; i++)
            grad[i] = nxt_grad[i].cwiseProduct(in[i].unaryExpr([](float x) -> float {
                float ex = std::exp(x);
                return ex * (x + ex + 1) / (ex + 1) / (ex + 1);
            }));
    }
};
struct mish : public layer {
    mish() : layer("mish") {}
    void forward(const vec_batch &in) {
        for (int i = 0; i < batch_sz; i++)
            out[i] = in[i].unaryExpr([](float x) -> float { return x * tanh(log(1 + exp(x))); });
    }
    void backward(const vec_batch &in, const vec_batch &nxt_grad) {
        for (int i = 0; i < batch_sz; i++)
            grad[i] = nxt_grad[i].cwiseProduct(in[i].unaryExpr([](float x) -> float {
                float ex = exp(x);
                float fm = (2 * ex + ex * ex + 2);
                return ex * (4 * (x + 1) + 4 * ex * ex + ex * ex * ex + ex * (4 * x + 6)) / fm / fm;
            }));
    }
};
struct softmax : public layer {
    softmax() : layer("softmax") {}
    void forward(const vec_batch &in) {
        for (int i = 0; i < batch_sz; i++) {
            out[i] = exp(in[i].array() - in[i].maxCoeff());
            out[i] /= out[i].sum();
        }
    }
    void backward(const vec_batch &in, const vec_batch &nxt_grad) { assert(0); }
};
struct same : public layer {
    same() : layer("same") {}
    void forward(const vec_batch &in) {
        for (int i = 0; i < batch_sz; i++) out[i] = in[i];
    }
    void backward(const vec_batch &in, const vec_batch &nxt_grad) {
        for (int i = 0; i < batch_sz; i++) grad[i] = nxt_grad[i];
    }
};
//}}}
//{{{ Layer to be trained
struct linear : public layer {
    const int in_sz, out_sz;
    // 系数，系数梯度
    mat weight, grad_weight;
    vec bias, grad_bias;
    linear(int in_sz, int out_sz)
        : layer("linear " + to_string(in_sz) + " -> " + to_string(out_sz))
        , in_sz(in_sz)
        , out_sz(out_sz)
        , weight(out_sz, in_sz)
        , grad_weight(out_sz, in_sz)
        , bias(out_sz)
        , grad_bias(out_sz) {
        bias.setZero();
        for (auto &i : weight.reshaped()) i = nd(0, sqrt(2. / in_sz));
    }
    void forward(const vec_batch &in) {
        for (int i = 0; i < batch_sz; i++) out[i] = weight * in[i] + bias;
    }
    void clear_grad() {
        grad_weight.setZero();
        grad_bias.setZero();
    }
    void backward(const vec_batch &in, const vec_batch &nxt_grad) {
        // nx.in==W*this.in+B
        // dnx.in/din
        for (int i = 0; i < batch_sz; i++) {
            grad_bias += nxt_grad[i];
            grad_weight += nxt_grad[i] * in[i].transpose();
            grad[i] = weight.transpose() * nxt_grad[i];
        }
        grad_weight /= batch_sz;
        grad_bias /= batch_sz;
    }
    void upd(optimizer &opt) {
        opt.upd(to_vecmap(weight), to_vecmap(grad_weight));
        opt.upd(to_vecmap(bias), to_vecmap(grad_bias));
    }
    void write(ostream &io) {
        for (auto &i : weight.reshaped()) io.write((char *)&i, sizeof(i));
        for (auto &i : bias.reshaped()) io.write((char *)&i, sizeof(i));
    }
    void read(istream &io) {
        for (auto &i : weight.reshaped()) io.read((char *)&i, sizeof(i));
        for (auto &i : bias.reshaped()) io.read((char *)&i, sizeof(i));
    }
};
struct batchnorm : public layer {
    // 平均值，方差
    vec mean, running_mean, grad_mean;
    vec var, running_var, grad_var, inv_var;
    vec gama, grad_gama;
    vec beta, grad_beta;
    // 这两个用来辅助,inv记录1/sqrt(方差+eps)
    vec grad_normalized_x;
    const float momentum;
    batchnorm(int sz, float momentum = 0.9)
        : layer("batchnorm " + to_string(sz))
        , mean(sz)
        , running_mean(sz)
        , grad_mean(sz)
        , var(sz)
        , running_var(sz)
        , grad_var(sz)
        , inv_var(sz)
        , gama(sz)
        , grad_gama(sz)
        , beta(sz)
        , grad_beta(sz)
        , momentum(momentum) {
        gama.setOnes();
        beta.setZero();
    }
    void forward(const vec_batch &in) {
        if (train_mode) {
            mean.setZero();
            var.setZero();
            for (int i = 0; i < batch_sz; i++) mean += in[i];
            mean /= batch_sz;
            for (int i = 0; i < batch_sz; i++) var += (in[i] - mean).cwiseAbs2();
            var /= batch_sz;
            inv_var = rsqrt(var.array() + EPS);
            running_mean = running_mean * momentum + mean * (1 - momentum);
            // 使用无偏方差
            // running_var = running_var * momentum + var * batch_sz / (batch_sz - 1) * (1 - momentum);
            running_var = running_var * momentum + var * (1 - momentum);

            for (int i = 0; i < batch_sz; i++)
                out[i] = (in[i] - mean).array() * inv_var.array() * gama.array() + beta.array();
        } else {
            for (int i = 0; i < batch_sz; i++)
                out[i] =
                    (in[i] - running_mean).array() * rsqrt(running_var.array() + EPS) * gama.array() + beta.array();
        }
    }
    void backward(const vec_batch &in, const vec_batch &nxt_grad) {
        for (int i = 0; i < batch_sz; i++) {
            grad_normalized_x = nxt_grad[i].array() * gama.array();

            grad_var.array() += grad_normalized_x.array() * (in[i] - mean).array();
            grad_mean.array() += grad_normalized_x.array();

            grad[i] = grad_normalized_x.array() * inv_var.array();

            grad_beta.array() += nxt_grad[i].array();
            grad_gama.array() += nxt_grad[i].array() * (in[i] - mean).array() * inv_var.array();
        }
        grad_var = -0.5 * grad_var.array() * inv_var.array().cube();
        grad_mean = -grad_mean.array() * inv_var.array();
        for (int i = 0; i < batch_sz; i++)
            grad[i].array() += (grad_mean.array() + 2 * grad_var.array() * (in[i] - mean).array()) / batch_sz;
        grad_beta /= batch_sz;
        grad_gama /= batch_sz;
    }
    void clear_grad() {
        grad_gama.setZero();
        grad_beta.setZero();
        grad_mean.setZero();
        grad_var.setZero();
    }
    void upd(optimizer &opt) {
        opt.upd(to_vecmap(beta), to_vecmap(grad_beta));
        opt.upd(to_vecmap(gama), to_vecmap(grad_gama));
    }
    void write(ostream &io) {
        for (auto &i : gama.reshaped()) io.write((char *)&i, sizeof(i));
        for (auto &i : beta.reshaped()) io.write((char *)&i, sizeof(i));
        for (auto &i : running_mean.reshaped()) io.write((char *)&i, sizeof(i));
        for (auto &i : running_var.reshaped()) io.write((char *)&i, sizeof(i));
    }
    void read(istream &io) {
        for (auto &i : gama.reshaped()) io.read((char *)&i, sizeof(i));
        for (auto &i : beta.reshaped()) io.read((char *)&i, sizeof(i));
        for (auto &i : running_mean.reshaped()) io.read((char *)&i, sizeof(i));
        for (auto &i : running_var.reshaped()) io.read((char *)&i, sizeof(i));
    }
};

ten3 hi_dim_conv(const ten3 &input, const ten4 &kernel, ten3 &res) {
    int sz1 = input.dimension(1) - kernel.dimension(2) + 1;
    int sz2 = input.dimension(2) - kernel.dimension(3) + 1;
    res = ten3(kernel.dimension(1), sz1, sz2);
    res.setZero();
    for (int i = 0; i < kernel.dimension(0); i++)
        for (int j = 0; j < kernel.dimension(1); j++)
            res.chip(j, 0) += input.chip(i, 0).convolve(kernel.chip(i, 0).chip(j, 0), Eigen::array<int, 2>{0, 1});
    return res;
}
struct conv : public layer {
    const int in_channel, out_channel, in_rows, in_cols;
    const int out_rows, out_cols;
    const int k_rows, k_cols;
    ten4 kernel, grad_kernel;
    vec bias, grad_bias;
    conv(int in_channel, int out_channel, int in_rows, int in_cols, int k_rows, int k_cols)
        : layer("conv " + to_string(in_channel) + " channels * " + to_string(in_rows) + " * " + to_string(in_cols) +
                " -> " + to_string(out_channel) + " channels * " + to_string(in_rows - k_rows + 1) + " * " +
                to_string(in_cols - k_cols + 1))
        , in_channel(in_channel)
        , out_channel(out_channel)
        , in_rows(in_rows)
        , in_cols(in_cols)
        , out_rows(in_rows - k_rows + 1)
        , out_cols(in_cols - k_cols + 1)
        , k_rows(k_rows)
        , k_cols(k_cols)
        , kernel(in_channel, out_channel, k_rows, k_cols)
        , grad_kernel(in_channel, out_channel, k_rows, k_cols)
        , bias(out_channel)
        , grad_bias(out_channel) {
        bias.setZero();
        for (int i = 0; i < in_channel; i++)
            for (int j = 0; j < out_channel; j++)
                for (int k = 0; k < k_rows; k++)
                    for (int l = 0; l < k_cols; l++)
                        kernel(i, j, k, l) = nd(0, sqrt(2. / (in_channel * in_rows * in_cols)));
    }
    void forward(const vec_batch &in) {
        for (int i = 0; i < batch_sz; i++) {
            vec tmp_vec = in[i];
            ten3 tensorout;
            hi_dim_conv(ten3map(tmp_vec.data(), in_channel, in_rows, in_cols), kernel, tensorout);
            for (int j = 0; j < out_channel; j++) tensorout.chip(j, 0) = tensorout.chip(j, 0) + bias[j];
            out[i] = to_vecmap(tensorout);
        }
    }
    void clear_grad() {
        grad_bias.setZero();
        grad_kernel.setZero();
    }
    void backward(const vec_batch &in, const vec_batch &nxt_grad) {
        for (int i = 0; i < batch_sz; i++) {
            vec _nxt_grad = nxt_grad[i], _in = in[i];
            grad[i].resize(in[0].size());
            ten3map grad_out(_nxt_grad.data(), out_channel, out_rows, out_cols);
            ten3map grad_in(grad[i].data(), in_channel, in_rows, in_cols);
            ten3map in_map(_in.data(), in_channel, in_rows, in_cols);
            grad_in.setZero();
            for (int j = 0; j < out_channel; j++) {
                grad_bias(j) += ten0(grad_out.chip(j, 0).sum())();
                for (int k = 0; k < in_channel; k++) {
                    // 转180°的卷积核
                    ten2 rot_kernel = kernel.chip(k, 0).chip(j, 0).reverse(Eigen::array<bool, 2>{true, true});
                    ten2 in_ten = grad_out.chip(j, 0).pad(Eigen::array<pair<int, int>, 2>{
                        pair<int, int>{k_rows - 1, k_rows - 1}, pair<int, int>{k_cols - 1, k_cols - 1}});
                    grad_in.chip(k, 0) += in_ten.convolve(rot_kernel, Eigen::array<int, 2>{0, 1});
                    // (i,j)--(k,l)-->(i-k,j-l)
                    grad_kernel.chip(k, 0).chip(j, 0) +=
                        in_map.chip(k, 0).convolve(grad_out.chip(j, 0), Eigen::array<int, 2>{0, 1});
                }
            }
        }
        grad_bias /= batch_sz;
        grad_kernel = grad_kernel / (float)batch_sz;
    }
    void upd(optimizer &opt) {
        opt.upd(to_vecmap(kernel), to_vecmap(grad_kernel));
        opt.upd(to_vecmap(bias), to_vecmap(grad_bias));
    }
    void write(ostream &io) {
        for (auto &i : bias.reshaped()) io.write((char *)&i, sizeof(i));
        for (int i = 0; i < in_channel; i++)
            for (int j = 0; j < out_channel; j++)
                for (int k = 0; k < k_rows; k++)
                    for (int l = 0; l < k_cols; l++) io.write((char *)&kernel(i, j, k, l), sizeof(kernel(i, j, k, l)));
    }
    void read(istream &io) {
        for (auto &i : bias.reshaped()) io.read((char *)&i, sizeof(i));
        for (int i = 0; i < in_channel; i++)
            for (int j = 0; j < out_channel; j++)
                for (int k = 0; k < k_rows; k++)
                    for (int l = 0; l < k_cols; l++) io.read((char *)&kernel(i, j, k, l), sizeof(kernel(i, j, k, l)));
    }
};

//}}}
//{{{ pooling

struct maxpool2x2 : public layer {
    const int in_channel, in_rows, in_cols;
    vector<Tensor<int, 3>> from;
    maxpool2x2(int in_channel, int in_rows, int in_cols)
        : layer("maxpool2x2"), in_channel(in_channel), in_rows(in_rows), in_cols(in_cols) {
        assert(in_rows % 2 == 0 && in_cols % 2 == 0);
    }
    void _resize(int new_batch_sz) { from.resize(new_batch_sz); }
    void forward(const vec_batch &in) {
        for (int i = 0; i < batch_sz; i++) {
            vec in_vec = in[i];
            ten3map in_ten(in_vec.data(), in_channel, in_rows, in_cols);
            ten3 res(in_channel, in_rows / 2, in_cols / 2);
            from[i] = Tensor<int, 3>(in_channel, in_rows / 2, in_cols / 2);
            for (int j = 0; j < in_channel; j++) {
                for (int k = 0; k < in_rows / 2; k++)
                    for (int l = 0; l < in_cols / 2; l++) {
                        res(j, k, l) = -INF;
                        if (res(j, k, l) < in_ten(j, k * 2, l * 2)) {
                            res(j, k, l) = in_ten(j, k * 2, l * 2);
                            from[i](j, k, l) = 0;
                        }
                        if (res(j, k, l) < in_ten(j, k * 2, l * 2 + 1)) {
                            res(j, k, l) = in_ten(j, k * 2, l * 2 + 1);
                            from[i](j, k, l) = 1;
                        }
                        if (res(j, k, l) < in_ten(j, k * 2 + 1, l * 2)) {
                            res(j, k, l) = in_ten(j, k * 2 + 1, l * 2);
                            from[i](j, k, l) = 2;
                        }
                        if (res(j, k, l) < in_ten(j, k * 2 + 1, l * 2 + 1)) {
                            res(j, k, l) = in_ten(j, k * 2 + 1, l * 2 + 1);
                            from[i](j, k, l) = 3;
                        }
                    }
            }
            out[i] = to_vecmap(res);
        }
    }
    void backward(const vec_batch &in, const vec_batch &nxt_grad) {
        for (int i = 0; i < batch_sz; i++) {
            vec nxt_vec = nxt_grad[i];
            ten3map nxt_ten(nxt_vec.data(), in_channel, in_rows / 2, in_cols / 2);
            ten3 res(in_channel, in_rows, in_cols);
            res.setZero();
            for (int j = 0; j < in_channel; j++) {
                for (int k = 0; k < in_rows / 2; k++)
                    for (int l = 0; l < in_cols / 2; l++) {
                        int &cur = from[i](j, k, l);
                        res(j, k * 2 + (cur >> 1), l * 2 + (cur & 1)) = nxt_ten(j, k, l);
                    }
            }
            grad[i] = to_vecmap(res);
        }
    }
};

//}}}
//{{{ Layers Sequence
struct net {
    virtual string shape() = 0;
    virtual void set_train_mode(const bool &) = 0;
    virtual vec_batch forward(const vec_batch &) = 0;
    virtual vec_batch backward(const vec_batch &) = 0;
    virtual void upd(optimizer &, const batch &) = 0;
    virtual void writef(const string &f) = 0;
    virtual void readf(const string &f) = 0;
    virtual vec_batch &out() = 0;
};
struct sequential : public net {
    int batch_sz;
    vector<layerp> layers;
    sequential() : batch_sz(0) {
        layerp x = make_shared<same>();
        x->name = "input";
        layers.emplace_back(x);
    }
    void add(const layerp &x) { layers.push_back(x); }
    string shape() {
        string res = "";
        for (auto &it : layers) res += it->name + "\n";
        return res;
    }
    void set_train_mode(const bool &new_train_mod) {
        for (auto &l : layers) l->set_train_mode(new_train_mod);
    }
    vec_batch forward(const vec_batch &input) {
        if ((int)input.size() != batch_sz) {
            batch_sz = input.size();
            for (auto &l : layers) l->resize(batch_sz);
        }
        int layer_sz = layers.size();
        layers[0]->forward(input);
        for (int i = 1; i < layer_sz; i++) layers[i]->forward(layers[i - 1]->out);
        return layers.back()->out;
    }
    vec_batch backward(const vec_batch &label) {
        for (int i = 0; i < batch_sz; i++) layers.back()->grad[i] = layers.back()->out[i] - label[i];
        int layer_sz = layers.size();
        for (int i = layer_sz - 2; i >= 0; i--)
            layers[i]->backward(i ? layers[i - 1]->out : vec_batch(), layers[i + 1]->grad);
        return layers[0]->grad;
    }
    void upd(optimizer &opt, const batch &data) {
        int layer_sz = layers.size();
        for (int i = 0; i < layer_sz; i++) layers[i]->clear_grad();
        forward(data.first);
        backward(data.second);
        for (int i = 0; i < layer_sz; i++) layers[i]->upd(opt);
    }
    void writef(const string &f) {
        ofstream fout(f, ios::binary | ios::out);
        int layer_sz = layers.size();
        for (int i = 0; i < layer_sz; i++) layers[i]->write(fout);
        fout.close();
    }
    void readf(const string &f) {
        ifstream fin(f, ios::binary | ios::in);
        int layer_sz = layers.size();
        for (int i = 0; i < layer_sz; i++) layers[i]->read(fin);
        fin.close();
    }
    vec_batch &out() { return layers.back()->out; }
};
//}}}
//{{{ Dataset
struct data_set {
    batch train, valid;
    data_set() {}
    data_set(const batch &all_data) {
        for (int i = 0; i < (int)all_data.first.size(); i++) {
            int rnd = ri(0, 6);
            if (rnd == 0) {
                valid.first.push_back(all_data.first[i]);
                valid.second.push_back(all_data.second[i]);
            } else {
                train.first.push_back(all_data.first[i]);
                train.second.push_back(all_data.second[i]);
            }
        }
    }
    batch get_train_batch(int batch_sz) const {
        assert(train.first.size());
        batch res;
        for (int i = 0; i < batch_sz; i++) {
            int id = ri(0, train.first.size() - 1);
            res.first.push_back(train.first[id]);
            res.second.push_back(train.second[id]);
        }
        return res;
    }
    batch get_valid_batch(int batch_sz) const {
        assert(valid.first.size());
        batch res;
        for (int i = 0; i < batch_sz; i++) {
            int id = ri(0, valid.first.size() - 1);
            res.first.push_back(valid.first[id]);
            res.second.push_back(valid.second[id]);
        }
        return res;
    }
};
//}}}
//{{{ Trainning
void upd(optimizer &opt, const data_set &data, net &net, int batch_sz, int epoch,
         function<float(const vec_batch &, const vec_batch &)> err_func, const string &save_file = "") {
    int t0 = clock();
    float tloss = 0, mult = 1, mn = INF;
    for (int i = 1; i <= epoch; i++) {
        auto tmp = data.get_train_batch(batch_sz);
        net.upd(opt, tmp);
        mult *= 0.9;
        tloss = tloss * 0.9 + err_func(net.out(), tmp.second) * 0.1;
        if (i % 50 == 0) {
            cerr << "-------------------------" << endl;
            cerr << "Time elapse: " << (float)(clock() - t0) / CLOCKS_PER_SEC << endl;
            cerr << "Epoch: " << i << endl;
            cerr << "Loss: " << tloss / (1. - mult) << endl;
            if (i % 1000 == 0) {
                net.set_train_mode(0);
                float sum = 0;
                for (int j = 0; j < (int)data.valid.first.size(); j++) {
                    batch tmp = {{data.valid.first[j]}, {data.valid.second[j]}};
                    sum += err_func(net.forward(tmp.first), tmp.second);
                }
                net.set_train_mode(1);
                sum /= data.valid.first.size();
                cerr << "!! Error: " << sum << endl;
                if (sum < mn && save_file != "") {
                    cerr << "Saved" << endl;
                    mn = sum;
                    net.writef(save_file);
                }
            }
        }
    }
}

//}}}

#endif
