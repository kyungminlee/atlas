#pragma once

#include <stdexcept>

namespace atlas {

struct cuda_exception : public std::exception
{
public:
    cuda_exception() noexcept
    : _message("") { }

    cuda_exception(char const * message) noexcept
    : _message(message) { }

    virtual char const * what() const noexcept override { return _message; }
private:
    char const * _message;
};

}