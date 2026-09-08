// Capture full teacher-forced logits with a llama.cpp-compatible C API.
//
// Input is written by qwen36_teacher.py capture-bf16. Output is contiguous
// float32 [prompts * teacher_steps, vocab] data for register-raw.

#include "llama.h"
#include "ggml-backend.h"
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

struct Capture {
    std::FILE * file = nullptr;
    uint32_t prompt = 0;
    uint32_t position = 0;
    bool active = false;
};

static bool capture_layer(ggml_tensor * tensor, bool ask, void * opaque) {
    const char * name = ggml_get_name(tensor);
    const char * prefixes[] = {"l_out-", "attn_norm-", "attn_residual-", "attn_post_norm-", "ffn_out-", "linear_attn_out-", "attn_output-", "final_output-", "state_predelta-", "conv_input-", "linear_attn_qkv_mixed-", "z-", "conv_output_silu-", "q_conv_predelta-", "k_conv_predelta-", "v_conv_predelta-", "gate-", "beta_sigmoid-", "result_norm", "result_output"};
    bool selected = false;
    for (auto prefix : prefixes) if (std::strncmp(name, prefix, std::strlen(prefix)) == 0) selected = true;
    if (ask) return selected;
    auto & state = *static_cast<Capture *>(opaque);
    if (!selected || !state.active) return true;
    if (tensor->type != GGML_TYPE_F32) {
        std::fprintf(stderr, "unexpected layer tensor layout: %s\n", name);
        std::abort();
    }
    const uint32_t n = static_cast<uint32_t>(ggml_nelements(tensor));
    const uint32_t size = static_cast<uint32_t>(std::strlen(name));
    float * data = static_cast<float *>(std::malloc(n * sizeof(float)));
    if (!data) std::abort();
    unsigned char * raw = static_cast<unsigned char *>(std::malloc(ggml_nbytes(tensor)));
    if (!raw) std::abort();
    ggml_backend_tensor_get(tensor, raw, 0, ggml_nbytes(tensor));
    size_t flat = 0;
    for (int64_t i3=0;i3<tensor->ne[3];++i3)
    for (int64_t i2=0;i2<tensor->ne[2];++i2)
    for (int64_t i1=0;i1<tensor->ne[1];++i1)
    for (int64_t i0=0;i0<tensor->ne[0];++i0) {
        std::memcpy(data+flat++, raw+i0*tensor->nb[0]+i1*tensor->nb[1]+i2*tensor->nb[2]+i3*tensor->nb[3], sizeof(float));
    }
    std::free(raw);
    const uint32_t header[] = {state.prompt, state.position, size, n};
    if (std::fwrite(header, sizeof(header), 1, state.file) != 1 ||
        std::fwrite(name, size, 1, state.file) != 1 ||
        std::fwrite(data, n * sizeof(float), 1, state.file) != 1) std::abort();
    std::free(data);
    return true;
}

#include <algorithm>
#include <fstream>
#include <string>
#include <vector>

struct Prompt {
    std::vector<llama_token> prompt;
    std::vector<llama_token> teacher;
};

static bool read_u32(std::ifstream & in, uint32_t * value) {
    in.read(reinterpret_cast<char *>(value), sizeof(*value));
    return static_cast<bool>(in);
}

static bool read_input(const std::string & path, std::vector<Prompt> * prompts) {
    std::ifstream in(path, std::ios::binary);
    char magic[4] = {};
    uint32_t version = 0;
    uint32_t count = 0;
    in.read(magic, sizeof(magic));
    if (!in || std::string(magic, sizeof(magic)) != "Q36Q" ||
            !read_u32(in, &version) || version != 1 || !read_u32(in, &count)) {
        return false;
    }
    prompts->clear();
    prompts->reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t prompt_count = 0;
        uint32_t teacher_count = 0;
        if (!read_u32(in, &prompt_count) || !read_u32(in, &teacher_count) ||
                prompt_count == 0 || teacher_count == 0) {
            return false;
        }
        Prompt row;
        row.prompt.resize(prompt_count);
        row.teacher.resize(teacher_count);
        in.read(reinterpret_cast<char *>(row.prompt.data()),
                static_cast<std::streamsize>(prompt_count * sizeof(int32_t)));
        in.read(reinterpret_cast<char *>(row.teacher.data()),
                static_cast<std::streamsize>(teacher_count * sizeof(int32_t)));
        if (!in) {
            return false;
        }
        prompts->push_back(std::move(row));
    }
    return true;
}

static void usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s MODEL INPUT.bin OUTPUT.f32 [THREADS=16] [GPU_LAYERS=0]\n",
        argv0);
}

int main(int argc, char ** argv) {
    if (argc < 4 || argc > 6) {
        usage(argv[0]);
        return 2;
    }
    const std::string model_path = argv[1];
    const std::string input_path = argv[2];
    const std::string output_path = argv[3];
    const int threads = argc >= 5 ? std::atoi(argv[4]) : 16;
    const int gpu_layers = argc >= 6 ? std::atoi(argv[5]) : 0;
    if (threads <= 0 || gpu_layers < 0) {
        usage(argv[0]);
        return 2;
    }

    std::vector<Prompt> prompts;
    if (!read_input(input_path, &prompts)) {
        std::fprintf(stderr, "failed to read teacher input: %s\n", input_path.c_str());
        return 3;
    }
    size_t max_tokens = 0;
    size_t output_rows = 0;
    for (const auto & prompt : prompts) {
        max_tokens = std::max(max_tokens, prompt.prompt.size() + prompt.teacher.size() - 1);
        output_rows += prompt.teacher.size();
    }

    llama_backend_init();
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = gpu_layers;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (model == nullptr) {
        std::fprintf(stderr, "failed to load model: %s\n", model_path.c_str());
        llama_backend_free();
        return 4;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t vocab_size = llama_vocab_n_tokens(vocab);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = static_cast<uint32_t>(std::max<size_t>(256, max_tokens + 1));
    cparams.n_batch = static_cast<uint32_t>(max_tokens);
    cparams.n_ubatch = static_cast<uint32_t>(max_tokens);
    cparams.n_threads = threads;
    cparams.n_threads_batch = threads;
    cparams.type_k = GGML_TYPE_BF16;
    cparams.type_v = GGML_TYPE_BF16;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    Capture capture;
    capture.file = std::fopen((output_path + ".layers").c_str(), "wb");
    if (!capture.file) return 9;
    cparams.cb_eval = capture_layer;
    cparams.cb_eval_user_data = &capture;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        std::fprintf(stderr, "failed to create context\n");
        llama_model_free(model);
        llama_backend_free();
        return 5;
    }

    std::ofstream output(output_path, std::ios::binary);
    if (!output) {
        std::fprintf(stderr, "failed to open output: %s\n", output_path.c_str());
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 6;
    }

    for (size_t prompt_index = 0; prompt_index < prompts.size(); ++prompt_index) {
        const Prompt & prompt = prompts[prompt_index];
        std::vector<llama_token> tokens = prompt.prompt;
        tokens.insert(tokens.end(), prompt.teacher.begin(), prompt.teacher.end() - 1);
        llama_memory_clear(llama_get_memory(ctx), true);

        llama_batch batch = llama_batch_init(1, 0, 1);
        batch.n_tokens = 1;
        const int32_t first_output = static_cast<int32_t>(prompt.prompt.size()) - 1;
        for (int32_t i = 0; i < static_cast<int32_t>(tokens.size()); ++i) {
            batch.token[0] = tokens[static_cast<size_t>(i)];
            batch.pos[0] = i;
            batch.n_seq_id[0] = 1;
            batch.seq_id[0][0] = 0;
            batch.logits[0] = i >= first_output ? 1 : 0;
            capture.prompt = static_cast<uint32_t>(prompt_index);
            capture.position = static_cast<uint32_t>(i);
            capture.active = i >= first_output && i <= first_output + 2;
            const int decode_rc = llama_decode(ctx, batch);
            if (decode_rc != 0) {
                std::fprintf(stderr, "decode failed prompt %zu position %d: %d\n", prompt_index, i, decode_rc);
                return 7;
            }
            llama_synchronize(ctx);
            if (i >= first_output) {
                const float * logits = llama_get_logits_ith(ctx, 0);
                if (logits == nullptr) return 8;
                output.write(reinterpret_cast<const char *>(logits),
                        static_cast<std::streamsize>(vocab_size * sizeof(float)));
            }
        }
        llama_batch_free(batch);
        output.flush();
        std::fprintf(stderr, "llama logits %2zu/%zu (%zu rows)\n",
                prompt_index + 1, prompts.size(), prompt.teacher.size());
    }

    output.close();
    std::fclose(capture.file);
    std::printf(
        "{\"prompts\":%zu,\"rows\":%zu,\"vocab_size\":%d,\"dtype\":\"float32\"}\n",
        prompts.size(), output_rows, vocab_size);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
