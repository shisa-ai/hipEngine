// Capture Muse Glimmer vision oracles from llama.cpp mtmd (read-only peer).
//
// Links against the external llama.cpp HIP build; not part of the hipEngine
// runtime. For each image argument it dumps:
//   <out>/<stem>.embd.f32  — projected rows [n_out, 6656] float32 LE
//   <out>/<stem>.json      — preprocessed size, chunk/token structure, sha256
//
// Build (from the repo root; peer build lives under ~/llama.cpp/llama.cpp-hip):
//   hipcc -O2 -std=c++17 scripts/muse_glimmer_vision_oracle.cpp \
//     -I~/llama.cpp/llama.cpp-hip/include \
//     -I~/llama.cpp/llama.cpp-hip/ggml/include \
//     -I~/llama.cpp/llama.cpp-hip/tools/mtmd \
//     -L~/llama.cpp/llama.cpp-hip/build/bin -lllama -lmtmd -lggml \
//     -Wl,-rpath,$HOME/llama.cpp/llama.cpp-hip/build/bin \
//     -o /tmp/muse_glimmer_vision_oracle
//
// Run:
//   /tmp/muse_glimmer_vision_oracle <text.gguf> <mmproj.gguf> <outdir> <img...>

#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <sys/stat.h>

// When GLIMMER_DUMP_TENSORS is set, capture named graph nodes during encode
// into <outdir>/tensors/<name>.f32 (+ .shape text) for layer-by-layer localization.
static std::string g_dump_dir;
static bool g_dump_enabled = false;

static bool dump_cb(struct ggml_tensor * t, bool ask, void * /*user*/) {
    if (!t->name[0]) return false;
    std::string name = t->name;
    static const char * prefixes[] = {
        "after_posemb", "after_sp_perm", "pre_ln", "after_inv_perm",
        "encoder_out", "projected",
        "layer_inp_normed-0", "Qcur-0", "Kcur-0", "Vcur-0",
        "Qcur_pos-0", "Kcur_pos-0", "attn_out-0", "ffn_inp-0",
        "ffn_inp_normed-0", "ffn_out-0", "layer_out-0",
        "layer_out-3", "layer_out-49",
    };
    bool want = false;
    for (const char * p : prefixes) if (name == p) { want = true; break; }
    if (!want) return false;
    if (ask) return true;
    std::string safe = name;
    for (auto & ch : safe) if (ch == '/') ch = '_';
    std::string path = g_dump_dir + "/" + safe + ".f32";
    FILE * f = fopen(path.c_str(), "wb");
    if (f) {
        // ggml tensors are contiguous in ne order for these nodes; write raw.
        fwrite(t->data, 1, ggml_nbytes(t), f);
        fclose(f);
        FILE * fs = fopen((path + ".shape").c_str(), "wb");
        fprintf(fs, "%lld %lld %lld %lld\n", (long long) t->ne[0], (long long) t->ne[1],
                (long long) t->ne[2], (long long) t->ne[3]);
        fclose(fs);
    }
    return true;
}

static std::string sha256_file(const char * path) {
    // reuse mtmd's bitmap id (it hashes the decoded pixels); for the file itself
    // we just record the bitmap id below. Placeholder kept simple.
    (void) path;
    return "";
}

int main(int argc, char ** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s <text.gguf> <mmproj.gguf> <outdir> <img...>\n", argv[0]);
        return 2;
    }
    const char * text_path = argv[1];
    const char * mm_path   = argv[2];
    const char * outdir    = argv[3];

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    llama_model * model = llama_model_load_from_file(text_path, mp);
    if (!model) { fprintf(stderr, "text model load failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_embd = llama_model_n_embd_inp(model);

    mtmd_context_params cp = mtmd_context_params_default();
    cp.use_gpu = getenv("GLIMMER_CPU") == nullptr;
    cp.warmup = false;
    cp.print_timings = false;
    cp.n_threads = 8;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    g_dump_enabled = getenv("GLIMMER_DUMP_TENSORS") != nullptr;
    if (g_dump_enabled) {
        g_dump_dir = std::string(outdir) + "/tensors";
        mkdir(g_dump_dir.c_str(), 0755);
        cp.cb_eval = dump_cb;
        cp.cb_eval_user_data = nullptr;
    }
    mtmd_context * ctx = mtmd_init_from_file(mm_path, model, cp);
    if (!ctx) { fprintf(stderr, "mtmd init failed\n"); return 1; }

    for (int ai = 4; ai < argc; ai++) {
        const char * img_path = argv[ai];
        std::string stem = img_path;
        auto slash = stem.find_last_of('/');
        if (slash != std::string::npos) stem = stem.substr(slash + 1);
        auto dot = stem.find_last_of('.');
        if (dot != std::string::npos) stem = stem.substr(0, dot);

        mtmd_helper_init_opt opt = mtmd_helper_init_opt_default();
        auto wrapper = mtmd_helper_bitmap_init_from_file(ctx, img_path, false, opt);
        if (!wrapper.bitmap) { fprintf(stderr, "bitmap load failed: %s\n", img_path); return 1; }
        const uint32_t nx = mtmd_bitmap_get_nx(wrapper.bitmap);
        const uint32_t ny = mtmd_bitmap_get_ny(wrapper.bitmap);

        const char * prompt = "Describe this image: <__media__>";
        mtmd_input_text text{prompt, strlen(prompt), true, true};
        const mtmd_bitmap * bitmaps[1] = {wrapper.bitmap};
        mtmd_input_chunks * chunks = mtmd_input_chunks_init();
        if (mtmd_tokenize(ctx, chunks, &text, bitmaps, 1) != 0) {
            fprintf(stderr, "tokenize failed\n"); return 1;
        }

        std::string json = "{\n";
        json += "  \"image\": \"" + std::string(img_path) + "\",\n";
        json += "  \"bitmap_id\": \"" + std::string(mtmd_bitmap_get_id(wrapper.bitmap)) + "\",\n";
        json += "  \"nx\": " + std::to_string(nx) + ",\n";
        json += "  \"ny\": " + std::to_string(ny) + ",\n";
        json += "  \"n_embd\": " + std::to_string(n_embd) + ",\n";
        json += "  \"chunks\": [\n";

        size_t n_out = 0;
        std::vector<float> embd_all;
        const size_t n_chunks = mtmd_input_chunks_size(chunks);
        for (size_t ci = 0; ci < n_chunks; ci++) {
            const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks, ci);
            auto type = mtmd_input_chunk_get_type(chunk);
            size_t nt = mtmd_input_chunk_get_n_tokens(chunk);
            json += "    {\"type\": \"";
            json += (type == MTMD_INPUT_CHUNK_TYPE_TEXT ? "text" :
                     type == MTMD_INPUT_CHUNK_TYPE_IMAGE ? "image" : "other");
            json += "\", \"n_tokens\": " + std::to_string(nt);
            if (type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
                size_t n = 0;
                const llama_token * toks = mtmd_input_chunk_get_tokens_text(chunk, &n);
                json += ", \"tokens\": [";
                for (size_t i = 0; i < n; i++) {
                    if (i) json += ", ";
                    json += std::to_string(toks[i]);
                }
                json += "]";
            } else if (type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
                if (mtmd_encode_chunk(ctx, chunk) != 0) {
                    fprintf(stderr, "encode failed\n"); return 1;
                }
                float * embd = mtmd_get_output_embd(ctx);
                embd_all.insert(embd_all.end(), embd, embd + nt * (size_t) n_embd);
                n_out += nt;
                auto pos = mtmd_image_tokens_get_decoder_pos(
                    mtmd_input_chunk_get_tokens_image(chunk), 0, 0);
                json += ", \"first_decoder_pos\": [" + std::to_string(pos.t) + ", " +
                        std::to_string(pos.x) + ", " + std::to_string(pos.y) + "]";
            }
            json += "}";
            if (ci + 1 < n_chunks) json += ",";
            json += "\n";
        }
        json += "  ],\n";
        json += "  \"n_out\": " + std::to_string(n_out) + "\n";
        json += "}\n";

        std::string jpath = std::string(outdir) + "/" + stem + ".json";
        std::string epath = std::string(outdir) + "/" + stem + ".embd.f32";
        FILE * fj = fopen(jpath.c_str(), "wb");
        fwrite(json.data(), 1, json.size(), fj);
        fclose(fj);
        FILE * fe = fopen(epath.c_str(), "wb");
        fwrite(embd_all.data(), sizeof(float), embd_all.size(), fe);
        fclose(fe);
        printf("%s: nx=%u ny=%u n_out=%zu -> %s\n", stem.c_str(), nx, ny, n_out, epath.c_str());

        mtmd_input_chunks_free(chunks);
        mtmd_bitmap_free(wrapper.bitmap);
    }

    mtmd_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}