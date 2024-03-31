/*
Copyright [2024] [Yao Yao]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

*/

//
// Created by yao on 12/11/17.
//

#pragma once
#include <eigen3/Eigen/Core>

namespace rsfm::legacy
{
template<typename T, int Size>
struct aligned_array{
    static_assert(Size > 0, "fatal error");
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    constexpr static unsigned size = Size;
    using data_type = Eigen::Array<T, Size, 1>;
    data_type data;

    aligned_array<T, Size> operator-() const{
        return aligned_array<T, Size>{-data};
    }

    aligned_array<T, Size>& operator=(const aligned_array<T, Size>& src){
        data = src.data;
        return *this;
    }

    aligned_array<T, Size>& operator+=(T val){
        data += val;
        return *this;
    }

    aligned_array<T, Size>& operator+=(const aligned_array<T, Size>& val){
        data += val.data;
        return *this;
    }

    aligned_array<T, Size>& operator-=(T val){
        data -= val;
        return *this;
    }

    aligned_array<T, Size>& operator-=(const aligned_array<T, Size>& val){
        data -= val.data;
        return *this;
    }

    aligned_array<T, Size>& operator*=(T val){
        data *= val;
        return *this;
    }

    aligned_array<T, Size>& operator*=(const aligned_array<T, Size>& val){
        data *= val.data;
        return *this;
    }

    aligned_array<T, Size> operator+(const T& val) const {
        return aligned_array<T, Size>(data + val);
    }

    aligned_array<T, Size> operator-(const T& val) const {
        return aligned_array<T, Size>(data - val);
    }

    aligned_array<T, Size> operator*(const T& val) const {
        return aligned_array<T, Size>(data * val);
    }

    aligned_array<T, Size> operator+(const aligned_array<T, Size>& val) const {
        return aligned_array<T, Size>{data + val.data};
    }

    aligned_array<T, Size> operator-(const aligned_array<T, Size>& val) const {
        return aligned_array<T, Size>{data - val.data};
    }

    aligned_array<T, Size> operator*(const aligned_array<T, Size>& val) const {
        return aligned_array<T, Size>{data * val.data};
    }

    aligned_array<T, Size> operator/(const aligned_array<T, Size>& val) const {
        return aligned_array<T, Size>{data / val.data};
    }

    T max() const{
        return data.maxCoeff();
    }

    T sum() const{
        return data.sum();
    }

    aligned_array() = default;

    aligned_array(const aligned_array<T, Size>& src){data = src.data;}

    template<typename Derived>
    explicit aligned_array(const Eigen::ArrayBase<Derived> & src){data = src;}

    explicit aligned_array(T val){
        data.setConstant(val);
    }

    static constexpr aligned_array zero() {return {data_type::Zero()};}
    static constexpr aligned_array one() {return {data_type::One()};}
    static constexpr aligned_array broadcast(T val) {data_type::One() * val;}
};

template<typename T, int Size>
inline const aligned_array<T, Size> operator+(const T& x,
                                              const aligned_array<T, Size>& y){
    return aligned_array<T, Size>{x + y.data};
}

template<typename T, int Size>
inline const aligned_array<T, Size> operator-(const T& x,
                                              const aligned_array<T, Size>& y){
    return aligned_array<T, Size>{x - y.data};
}

template<typename T, int Size>
inline const aligned_array<T, Size> operator*(const T& x,
                                              const aligned_array<T, Size>& y){
    return aligned_array<T, Size>{x * y.data};
}

template<typename T, int Size>
inline const aligned_array<T, Size> operator/(const T& x,
                                              const aligned_array<T, Size>& y){
    return aligned_array<T, Size>{x / y.data};
}

template<typename T, int Size>
inline const aligned_array<T, Size> conj(const aligned_array<T, Size>& x)  { return x; }

template<typename T, int Size>
inline const aligned_array<T, Size> real(const aligned_array<T, Size>& x)  { return x; }

template<typename T, int Size>
inline aligned_array<T, Size> imag(const aligned_array<T, Size>& x)    {
    return aligned_array<T, Size>{Eigen::Array<T, Size, 1>::Zero(x.data.rows())};
}

template<typename T, int Size>
inline aligned_array<T, Size> abs(const aligned_array<T, Size>&  x)  { return aligned_array<T, Size>{x.data.abs()}; }

template<typename T, int Size>
inline aligned_array<T, Size> abs2(const aligned_array<T, Size>& x) { return x * x;};

} // namespace rsfm::legacy

namespace Eigen {
    template<typename T, int Size>
    struct NumTraits<rsfm::legacy::aligned_array<T, Size>> : GenericNumTraits<rsfm::legacy::aligned_array<T, Size>>
    {
        typedef rsfm::legacy::aligned_array<T, Size> Real;
        typedef rsfm::legacy::aligned_array<T, Size> NonInteger;
        typedef rsfm::legacy::aligned_array<T, Size> Nested;

        static inline Real epsilon() { return Real(0); }
        static inline Real dummy_precision() { return Real(0); }
        static inline Real digits10() { return Real(0); }

        enum {
            IsComplex = 0,
            IsInteger = 0,
            IsSigned = 1,
            RequireInitialization = 1,
            ReadCost = 1,
            AddCost = 3,
            MulCost = 3
        };
    };
}
