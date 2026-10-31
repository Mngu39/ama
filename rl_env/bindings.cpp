#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "puyo_env.h"
#include <string>
#include <vector>

namespace py = pybind11;

// Convert Ama's C++ observation into Python data.
static py::dict to_python(const PuyoEnv::Observation& obs) {
    py::dict result;

    py::list board;
    for (const auto& row : obs.board) {
        board.append(std::string(row.begin(), row.end()));
    }
    py::list visible;
    for (const auto& pair : obs.visible) {
        visible.append(py::make_tuple(
            std::string(1, cell::to_char(pair.first)),
            std::string(1, cell::to_char(pair.second))
        ));
    }

    result["board"] = board;
    result["row14"] = obs.row14;
    result["visible"] = visible;
    result["turn"] = obs.turn;

    return result;
}

static py::dict to_python_result(const PuyoEnv::Result& r) {
    py::dict result;
    result["observation"] = to_python(r.observation);
    result["chains"] = r.chains;
    result["score"] = r.score;
    result["terminated"] = r.terminated;
    result["truncated"] = r.truncated;
    return result;
}

PYBIND11_MODULE(puyo_cpp, m) {
    m.doc() = "Ama Puyo Puyo RL environment";
    m.attr("ACTION_COUNT") = PuyoEnv::ACTION_COUNT;
    m.attr("MAX_TURNS") = PuyoEnv::MAX_TURNS;

    py::class_<PuyoEnv>(m, "PuyoEnv")
        .def(py::init<>())

        .def("reset", [](PuyoEnv& env, u32 seed) {
            return to_python(env.reset(seed));
        }, py::arg("seed"))

        .def("observation", [](PuyoEnv& env) {
            return to_python(env.observation());
        })
        .def("legal_actions", [](PuyoEnv& env) {
            auto mask = env.legal_actions();
            return std::vector<bool>(mask.begin(), mask.end());
        })

        .def("step", [](PuyoEnv& env, int action) {
            return to_python_result(env.step(action));
        }, py::arg("action"))

        .def("preview", [](const PuyoEnv& env, int action) {
            return to_python_result(env.preview(action));
        }, py::arg("action"))

        .def("preview_all", [](PuyoEnv& env) {
            py::list result;
            auto mask = env.legal_actions();
            for (int action = 0; action < PuyoEnv::ACTION_COUNT; ++action) {
                if (!mask[action]) continue;
                py::dict candidate = to_python_result(env.preview(action));
                candidate["action"] = action;
                result.append(candidate);
            }
            return result;
        })

        .def("done", &PuyoEnv::done);
}
