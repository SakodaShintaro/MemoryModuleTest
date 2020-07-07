#ifndef DOUBLE_OSTREAM_HPP
#define DOUBLE_OSTREAM_HPP

//標準出力とファイルストリームに同時に出力するためのクラス
//参考)https://aki-yam.hatenablog.com/entry/20080630/1214801872
class DoubleOstream {
public:
    explicit DoubleOstream(std::ostream &_os1, std::ostream &_os2) : os1(_os1), os2(_os2) {};
    template <typename T>
    DoubleOstream& operator<< (const T &rhs)  { os1 << rhs;  os2 << rhs; return *this; };
    DoubleOstream& operator<< (std::ostream& (*__pf)(std::ostream&))  { __pf(os1); __pf(os2); return *this; };
    /*!<  Interface for manipulators, such as \c std::endl and \c std::setw
      For more information, see ostream header */

private:
    std::ostream& os1;
    std::ostream& os2;
};

#endif //DOUBLE_OSTREAM_HPP