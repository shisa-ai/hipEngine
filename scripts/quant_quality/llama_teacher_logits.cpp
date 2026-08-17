// Capture full teacher-forced logits with a llama.cpp-compatible C API.
//
// Input is written by qwen36_teacher.py capture-bf16. Output is contiguous
// float32 [prompts * teacher_steps, vocab] data for register-raw.

#include "llama.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
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

        llama_batch batch = llama_batch_init(static_cast<int32_t>(tokens.size()), 0, 1);
        batch.n_tokens = static_cast<int32_t>(tokens.size());
        const int32_t first_output = static_cast<int32_t>(prompt.prompt.size()) - 1;
        for (int32_t i = 0; i < batch.n_tokens; ++i) {
            batch.token[i] = tokens[static_cast<size_t>(i)];
            batch.pos[i] = i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = i >= first_output ? 1 : 0;
        }
        const int decode_rc = llama_decode(ctx, batch);
        llama_batch_free(batch);
        if (decode_rc != 0) {
            std::fprintf(stderr, "llama_decode failed for prompt %zu: %d\n", prompt_index, decode_rc);
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return 7;
        }
        llama_synchronize(ctx);
        for (size_t row = 0; row < prompt.teacher.size(); ++row) {
            const float * logits = llama_get_logits_ith(
                    ctx, first_output + static_cast<int32_t>(row));
            if (logits == nullptr) {
                std::fprintf(stderr, "missing logits for prompt %zu row %zu\n", prompt_index, row);
                llama_free(ctx);
                llama_model_free(model);
                llama_backend_free();
                return 8;
            }
            output.write(reinterpret_cast<const char *>(logits),
                    static_cast<std::streamsize>(vocab_size * sizeof(float)));
        }
        output.flush();
        std::fprintf(stderr, "llama logits %2zu/%zu (%zu rows)\n",
                prompt_index + 1, prompts.size(), prompt.teacher.size());
    }

    output.close();
    std::printf(
        "{\"prompts\":%zu,\"rows\":%zu,\"vocab_size\":%d,\"dtype\":\"float32\"}\n",
        prompts.size(), output_rows, vocab_size);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
