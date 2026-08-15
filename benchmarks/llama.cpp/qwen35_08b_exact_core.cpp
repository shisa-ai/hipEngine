// D08 closure helper for llama.cpp's exact-token no-sampler core scope.
//
// Build this file against matched read-only llama.cpp HIP and Vulkan checkouts.
// The prompt and teacher token/count arguments come from
// benchmarks/fixtures/qwen35_08b_vulkan_parity_p512_t128.json. Core throughput
// excludes top-1 scans; public throughput includes greedy scans and feedback.
// Separate untimed repeats record deterministic top-1 for both scopes.

#include "llama.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

using clock_type = std::chrono::steady_clock;

static double ms_since(clock_type::time_point start) {
    return std::chrono::duration<double, std::milli>(
               clock_type::now() - start)
        .count();
}

static int top1(llama_context * ctx, int vocab_size, bool & finite) {
    float * logits = llama_get_logits_ith(ctx, -1);
    if (logits == nullptr) {
        std::cerr << "missing logits\n";
        std::exit(4);
    }
    int best = 0;
    for (int index = 0; index < vocab_size; ++index) {
        finite = finite && std::isfinite(logits[index]);
        if (logits[index] > logits[best]) {
            best = index;
        }
    }
    return best;
}

static void decode_or_die(
    llama_context * ctx,
    llama_token * tokens,
    int count
) {
    llama_batch batch = llama_batch_get_one(tokens, count);
    const int status = llama_decode(ctx, batch);
    if (status != 0) {
        std::cerr << "llama_decode failed: " << status << "\n";
        std::exit(3);
    }
}

int main(int argc, char ** argv) {
    if (argc != 7) {
        std::cerr
            << "usage: qwen35_08b_exact_core MODEL PROMPT_ID PROMPT_TOKENS "
            << "TEACHER_ID FORCED_TOKENS REPS\n";
        return 2;
    }
    const char * model_path = argv[1];
    const char * engine_label = std::getenv("QWEN35_08B_ENGINE_LABEL");
    if (engine_label == nullptr || engine_label[0] == '\0') {
        engine_label = "llamacpp";
    }
    const int prompt_id = std::stoi(argv[2]);
    const int prompt_tokens = std::stoi(argv[3]);
    const int teacher_id = std::stoi(argv[4]);
    const int forced_tokens = std::stoi(argv[5]);
    const int repetitions = std::stoi(argv[6]);
    if (
        prompt_id < 0 || teacher_id < 0 || prompt_tokens <= 0 ||
        forced_tokens <= 0 || repetitions <= 0
    ) {
        std::cerr << "token IDs must be non-negative and counts positive\n";
        return 2;
    }

    ggml_backend_load_all();
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 99;
    llama_model * model = llama_model_load_from_file(model_path, model_params);
    if (model == nullptr) {
        return 3;
    }

    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = prompt_tokens + forced_tokens + 8;
    context_params.n_batch = prompt_tokens;
    context_params.n_ubatch = prompt_tokens;
    context_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    context_params.no_perf = true;
    llama_context * context = llama_init_from_model(model, context_params);
    if (context == nullptr) {
        llama_model_free(model);
        return 3;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int vocab_size = llama_vocab_n_tokens(vocab);
    std::vector<llama_token> prompt(prompt_tokens, prompt_id);
    llama_token teacher = teacher_id;

    auto run = [&](bool collect, std::vector<int> * top1_ids) {
        llama_memory_clear(llama_get_memory(context), true);
        const auto prompt_start = clock_type::now();
        decode_or_die(context, prompt.data(), prompt_tokens);
        llama_synchronize(context);
        const double prompt_ms = ms_since(prompt_start);
        bool finite = true;
        if (collect) {
            top1_ids->push_back(top1(context, vocab_size, finite));
        }
        const auto decode_start = clock_type::now();
        for (int index = 0; index < forced_tokens; ++index) {
            decode_or_die(context, &teacher, 1);
            if (collect) {
                top1_ids->push_back(top1(context, vocab_size, finite));
            }
        }
        llama_synchronize(context);
        const double decode_ms = ms_since(decode_start);
        if (!finite) {
            std::cerr << "non-finite logits\n";
            std::exit(5);
        }
        return std::pair<double, double>(prompt_ms, decode_ms);
    };

    auto run_public = [&](bool collect, std::vector<int> * top1_ids) {
        llama_memory_clear(llama_get_memory(context), true);
        const auto prompt_start = clock_type::now();
        decode_or_die(context, prompt.data(), prompt_tokens);
        bool finite = true;
        llama_token current = top1(context, vocab_size, finite);
        if (collect) {
            top1_ids->push_back(current);
        }
        const double prompt_ms = ms_since(prompt_start);
        const auto decode_start = clock_type::now();
        for (int index = 0; index < forced_tokens; ++index) {
            decode_or_die(context, &current, 1);
            current = top1(context, vocab_size, finite);
            if (collect) {
                top1_ids->push_back(current);
            }
        }
        const double decode_ms = ms_since(decode_start);
        if (!finite) {
            std::cerr << "non-finite public logits\n";
            std::exit(5);
        }
        return std::pair<double, double>(prompt_ms, decode_ms);
    };

    run(false, nullptr);
    run_public(false, nullptr);
    std::vector<double> prompt_ms;
    std::vector<double> decode_ms;
    std::vector<double> public_prompt_ms;
    std::vector<double> public_decode_ms;
    for (int index = 0; index < repetitions; ++index) {
        const auto row = run(false, nullptr);
        prompt_ms.push_back(row.first);
        decode_ms.push_back(row.second);
        const auto public_row = run_public(false, nullptr);
        public_prompt_ms.push_back(public_row.first);
        public_decode_ms.push_back(public_row.second);
    }
    std::vector<int> top1_first;
    std::vector<int> top1_second;
    std::vector<int> public_top1_first;
    std::vector<int> public_top1_second;
    run(true, &top1_first);
    run(true, &top1_second);
    run_public(true, &public_top1_first);
    run_public(true, &public_top1_second);
    const bool deterministic = top1_first == top1_second;
    const bool public_deterministic = public_top1_first == public_top1_second;

    std::cout << "{\n  \"schema\":1,\n  \"engine\":\"" << engine_label << "\",\n"
              << "  \"model\":\"" << model_path << "\",\n"
              << "  \"prompt_token_id\":" << prompt_id << ",\n"
              << "  \"prompt_tokens\":" << prompt_tokens << ",\n"
              << "  \"teacher_token_id\":" << teacher_id << ",\n"
              << "  \"forced_tokens\":" << forced_tokens << ",\n"
              << "  \"repetitions\":" << repetitions << ",\n"
              << "  \"prefill_ms\":[";
    for (int index = 0; index < repetitions; ++index) {
        if (index) std::cout << ",";
        std::cout << prompt_ms[index];
    }
    std::cout << "],\n  \"decode_ms\":[";
    for (int index = 0; index < repetitions; ++index) {
        if (index) std::cout << ",";
        std::cout << decode_ms[index];
    }
    std::cout << "],\n  \"public_prefill_ms\":[";
    for (int index = 0; index < repetitions; ++index) {
        if (index) std::cout << ",";
        std::cout << public_prompt_ms[index];
    }
    std::cout << "],\n  \"public_decode_ms\":[";
    for (int index = 0; index < repetitions; ++index) {
        if (index) std::cout << ",";
        std::cout << public_decode_ms[index];
    }
    std::cout << "],\n  \"top1_ids\":[";
    for (size_t index = 0; index < top1_first.size(); ++index) {
        if (index) std::cout << ",";
        std::cout << top1_first[index];
    }
    std::cout << "],\n  \"top1_deterministic\":"
              << (deterministic ? "true" : "false")
              << ",\n  \"public_top1_ids\":[";
    for (size_t index = 0; index < public_top1_first.size(); ++index) {
        if (index) std::cout << ",";
        std::cout << public_top1_first[index];
    }
    std::cout << "],\n  \"public_top1_deterministic\":"
              << (public_deterministic ? "true" : "false") << "\n}\n";

    llama_free(context);
    llama_model_free(model);
    return deterministic && public_deterministic ? 0 : 6;
}
