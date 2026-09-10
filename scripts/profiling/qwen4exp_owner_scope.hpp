#pragma once

#include "ggml.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_set>

// Profiling metadata only: never change tensor names, operands or op parameters.
namespace hipengine_profile {
inline std::unordered_set<const ggml_tensor *> & tagged_nodes() {
    static thread_local std::unordered_set<const ggml_tensor *> nodes;
    return nodes;
}

inline std::string hex_name(const char * text) {
    static const char digits[] = "0123456789abcdef";
    std::string result;
    for (const unsigned char * p = reinterpret_cast<const unsigned char *>(text); *p; ++p) {
        result += digits[*p >> 4];
        result += digits[*p & 15];
    }
    return result.empty() ? "-" : result;
}

struct owner_scope {
    ggml_context * ctx;
    ggml_tensor * last = nullptr;
    const char * owner;
    bool enabled;

    owner_scope(ggml_context * context, const char * family, bool root = false)
        : ctx(context), owner(family), enabled(std::getenv("HIPENGINE_VK_OWNER_TRACE") != nullptr) {
        if (!enabled) return;
        if (root) {
            tagged_nodes().clear();
            std::fprintf(stderr, "HE_OWNER_GRAPH_BEGIN\n");
        }
        for (auto * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) last = t;
    }

    ~owner_scope() {
        if (!enabled) return;
        auto * t = last ? ggml_get_next_tensor(ctx, last) : ggml_get_first_tensor(ctx);
        for (; t; t = ggml_get_next_tensor(ctx, t)) {
            if (!tagged_nodes().insert(t).second) continue;
            const char * family = owner;
            const char * weight = t->src[0] ? t->src[0]->name : "";
            if (t->op == GGML_OP_MUL_MAT_ID) {
                family = "moe";
            } else if (t->op == GGML_OP_MUL_MAT) {
                const bool router = std::strstr(weight, "ffn_gate_inp") != nullptr;
                const bool gr_up = std::strcmp(owner, "gr_read") == 0 &&
                    (std::strstr(weight, "hc_attn_up") || std::strstr(weight, "hc_ffn_up"));
                family = router ? "moe" : gr_up ? "gr_read" : "linear";
            }
            const auto name = hex_name(t->name);
            const auto source = hex_name(weight);
            std::fprintf(stderr, "HE_OWNER %p %s %s %s %s\n",
                         static_cast<void *>(t), family, ggml_op_name(t->op),
                         name.c_str(), source.c_str());
        }
    }
};
}
