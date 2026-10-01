#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "block_pool.h"
namespace py = pybind11;
PYBIND11_MODULE(_allocator, m) {
  py::class_<paged::Append>(m,"Append")
    .def_readonly("slots", &paged::Append::slots)
    .def_readonly("copies", &paged::Append::copies);
  py::class_<paged::BlockPool>(m,"BlockPool")
    .def(py::init<int,int,int>())
    .def("create", &paged::BlockPool::create)
    .def("fork", &paged::BlockPool::fork)
    .def("release", &paged::BlockPool::release)
    .def("append", &paged::BlockPool::append)
    .def("length", [](const paged::BlockPool& p,int id){return p.sequence(id).length;})
    .def("table", [](const paged::BlockPool& p,int id){return p.sequence(id).blocks;})
    .def("refcount", &paged::BlockPool::refcount)
    .def_property_readonly("free_blocks", &paged::BlockPool::free_blocks);
}
