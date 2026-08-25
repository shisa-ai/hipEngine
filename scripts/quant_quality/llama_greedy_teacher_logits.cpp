// Generate greedy BF16-teacher tokens and full logits through llama.cpp.
// Input: Q38Q v1 prompt-only token rows. Outputs raw float32 logits plus one
// comma-separated teacher-token row per prompt.

#include "llama.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

static bool read_u32(std::ifstream & in, uint32_t * value) {
    in.read(reinterpret_cast<char *>(value), sizeof(*value));
    return static_cast<bool>(in);
}

static bool read_input(const std::string & path, std::vector<std::vector<llama_token>> * prompts) {
    std::ifstream in(path, std::ios::binary);
    char magic[4] = {};
    uint32_t version = 0;
    uint32_t count = 0;
    in.read(magic, sizeof(magic));
    if (!in || std::string(magic, sizeof(magic)) != "Q38Q" ||
            !read_u32(in, &version) || version != 1 || !read_u32(in, &count)) {
        return false;
    }
    prompts->clear();
    prompts->reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t token_count = 0;
        if (!read_u32(in, &token_count) || token_count == 0) {
            return false;
        }
        std::vector<llama_token> prompt(token_count);
        in.read(reinterpret_cast<char *>(prompt.data()),
                static_cast<std::streamsize>(token_count * sizeof(int32_t)));
        if (!in) {
            return false;
        }
        prompts->push_back(std::move(prompt));
    }
    return true;
}

static llama_token argmax_token(const float * logits, int32_t vocab_size) {
    llama_token result = 0;
    float best = -std::numeric_limits<float>::infinity();
    for (int32_t token = 0; token < vocab_size; ++token) {
        if (logits[token] > best) {
            best = logits[token];
            result = token;
        }
    }
    return result;
}

static void usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s MODEL PROMPTS.bin LOGITS.f32 TOKENS.txt STEPS [THREADS=16] [GPU_LAYERS=999]\n",
        argv0);
}

int main(int argc, char ** argv) {
    if (argc < 6 || argc > 8) {
        usage(argv[0]);
        return 2;
    }
    const std::string model_path = argv[1];
    const std::string prompt_path = argv[2];
    const std::string logits_path = argv[3];
    const std::string tokens_path = argv[4];
    const int steps = std::atoi(argv[5]);
    const int threads = argc >= 7 ? std::atoi(argv[6]) : 16;
    const int gpu_layers = argc >= 8 ? std::atoi(argv[7]) : 999;
    if (steps <= 0 || threads <= 0 || gpu_layers < 0) {
        usage(argv[0]);
        return 2;
    }

    std::vector<std::vector<llama_token>> prompts;
    if (!read_input(prompt_path, &prompts)) {
        std::fprintf(stderr, "failed to read prompt input: %s\n", prompt_path.c_str());
        return 3;
    }
    size_t max_tokens = 0;
    for (const auto & prompt : prompts) {
        max_tokens = std::max(max_tokens, prompt.size() + static_cast<size_t>(steps));
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

    std::ofstream logits_output(logits_path, std::ios::binary);
    std::ofstream tokens_output(tokens_path);
    if (!logits_output || !tokens_output) {
        std::fprintf(stderr, "failed to open outputs\n");
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 6;
    }

    for (size_t prompt_index = 0; prompt_index < prompts.size(); ++prompt_index) {
        const auto & prompt = prompts[prompt_index];
        llama_memory_clear(llama_get_memory(ctx), true);
        llama_batch batch = llama_batch_init(static_cast<int32_t>(prompt.size()), 0, 1);
        batch.n_tokens = static_cast<int32_t>(prompt.size());
        for (int32_t i = 0; i < batch.n_tokens; ++i) {
            batch.token[i] = prompt[static_cast<size_t>(i)];
            batch.pos[i] = i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = i + 1 == batch.n_tokens ? 1 : 0;
        }
        int rc = llama_decode(ctx, batch);
        llama_batch_free(batch);
        if (rc != 0) {
            std::fprintf(stderr, "prompt decode failed for %zu: %d\n", prompt_index, rc);
            return 7;
        }

        llama_token current = 0;
        for (int step = 0; step < steps; ++step) {
            llama_synchronize(ctx);
            const float * logits = llama_get_logits_ith(ctx, -1);
            if (logits == nullptr) {
                std::fprintf(stderr, "missing logits for prompt %zu step %d\n", prompt_index, step);
                return 8;
            }
            logits_output.write(reinterpret_cast<const char *>(logits),
                    static_cast<std::streamsize>(vocab_size * sizeof(float)));
            current = argmax_token(logits, vocab_size);
            if (step != 0) {
                tokens_output << ',';
            }
            tokens_output << current;
            if (step + 1 < steps) {
                llama_batch next = llama_batch_init(1, 0, 1);
                next.n_tokens = 1;
                next.token[0] = current;
                next.pos[0] = static_cast<int32_t>(prompt.size()) + step;
                next.n_seq_id[0] = 1;
                next.seq_id[0][0] = 0;
                next.logits[0] = 1;
                rc = llama_decode(ctx, next);
                llama_batch_free(next);
                if (rc != 0) {
                    std::fprintf(stderr, "step decode failed for %zu step %d: %d\n", prompt_index, step, rc);
                    return 9;
                }
            }
        }
        tokens_output << '\n';
        logits_output.flush();
        tokens_output.flush();
        std::fprintf(stderr, "BF16 teacher %2zu/%zu\n", prompt_index + 1, prompts.size());
    }

    std::printf("{\"prompts\":%zu,\"rows\":%zu,\"vocab_size\":%d}\n",
            prompts.size(), prompts.size() * static_cast<size_t>(steps), vocab_size);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
