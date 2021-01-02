#ifndef VEDASDEVICEVECTOR_H
#define VEDASDEVICEVECTOR_H

#include <moderngpu/context.hxx>
#include "vedas.h"

class VedasDeviceVector
{
public:
    VedasDeviceVector();
    TYPEID *getCppArrayPtr();
private:
    TYPEID *vec;
    size_t n;
};

#endif // VEDASDEVICEVECTOR_H
